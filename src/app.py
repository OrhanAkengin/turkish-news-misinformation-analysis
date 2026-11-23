from pathlib import Path
import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "calibrated_svm_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "calibrated_svm_vectorizer.pkl"


# Kriter tabanlı analizde kullanılacak anahtar ifadeler.
# Bunlar kesin kanıt değil, risk sinyali olarak değerlendirilir.
CRITERIA_KEYWORDS = {
    "Kaynak Eksikliği": [
        "kaynak belirtilmedi",
        "resmi kaynak yok",
        "kaynak yok",
        "kanıt yok",
        "belge yok",
        "iddia edildi",
        "öne sürüldü",
        "sosyal medyada yayıldı",
        "sosyal medyada paylaşıldı",
        "görgü tanıkları",
        "isimsiz kaynak",
        "uzmanlar açıkladı",
        "gizli uzmanlar",
        "araştırmalara göre",
        "bilim insanları söylüyor",
        "resmi kurum açıklamasına yer verilmedi",
        "banka duyurusuna yer verilmedi",
        "güvenilir haber kaynağına yer verilmedi",
        "resmi açıklama yok",
        "duyuru yok",
        "doğrulanabilir bilgiye yer verilmedi",
        "herhangi bir doğrulanabilir bilgiye yer verilmedi",
        "kaynağına yer verilmedi",
        "tarihine yer verilmedi",
        "resmi kurum açıklamasına yer verilmedi",
        "bilimsel araştırmaya yer verilmedi",
        "uzman görüşüne yer verilmedi",
        "resmi sağlık kurumu açıklamasına yer verilmedi",
        "bilimsel rapora yer verilmedi",
        "güvenilir haber kaynağına yer verilmedi",
        "iddialara göre",
        "iddialara gore",
        "iddia var",
        "doğrulanmadı",
        "dogrulanmadi",
        "henüz doğrulanmadı",
        "henüz doğrulanmış değil",
        "resmi açıklama yapılmadı",
        "resmi kaynaklardan doğrulanmadı",
        "teyit edilmedi",
        "tahminlere göre",
        "tahminlere gore",


    ],
    "Kışkırtıcı Dil": [
        "son dakika",
        "şok",
        "flaş",
        "skandal",
        "gizli",
        "saklanan gerçek",
        "herkes bunu konuşuyor",
        "ortaya çıktı",
        "büyük panik",
        "korkunç",
        "tehlike",
        "uyarıldı",
        "paylaşmadan geçme",
        "öldü",
        "hayatını kaybetti",
        "vefat etti",
        "suikast",
        "darbe",
        "ülke karıştı",
        "acil",
        "kriz",
        "kaos",
    ],
    "Duygu Aşırılığı": [
        "mucize",
        "inanılmaz",
        "kesin çözüm",
        "hemen",
        "asla",
        "tamamen",
        "tüm hastalıkları",
        "bir gecede",
        "birkaç saat içinde",
        "hayat kurtarıyor",
        "ölümcül",
        "dehşet",
        "panik",
        "herkes yapmalı",
        "öldü",
        "herkes panikte",
        "ülke ayağa kalktı",
        "büyük şok",
        "acil paylaşın",
    ],
    "Hatalı Bağlam": [
    "eski görüntü",
    "eski bir görüntü",
    "eski video",
    "eski bir video",
    "bağlamından koparıldı",
    "bağlamına ilişkin",
    "olayın bağlamına",
    "yanlış tarihte",
    "yanlış tarih",
    "tarihine ilişkin",
    "yanlış yerde",
    "farklı olay",
    "başka bir olay",
    "başka bir olaydan",
    "farklı bir ülkede",
    "farklı ülkede",
    "videonun kaynağı",
    "görüntülerin",
    "paylaşılan görüntülerin",
    "montaj",
    "kesilmiş görüntü",
    "çarpıtıldı",
    "manipüle edildi",
],
}


# İlk aşamada BWM/anket ağırlıklarını elle temsil ediyoruz.
# Sonraki adımda bunu anket dosyasından otomatik hesaplayacağız.
# Toplamları 1 olmalı.
CRITERIA_WEIGHTS = {
    "Kaynak Eksikliği": 0.332090,
    "Hatalı Bağlam": 0.231343,
    "Duygu Aşırılığı": 0.227612,
    "Kışkırtıcı Dil": 0.208955,
}



def risk_level(score):
    """
    0-100 arasındaki risk skorunu sözel risk seviyesine çevirir.
    """

    if score < 30:
        return "Düşük Risk"
    elif score < 60:
        return "Orta Risk"
    else:
        return "Yüksek Risk"


def risk_explanation(score):
    """
    Nihai risk skoruna göre kısa açıklama üretir.
    """

    if score < 30:
        return (
            "Girilen metin, sistem tarafından düşük dezenformasyon riski taşıyan "
            "bir içerik olarak değerlendirilmiştir. Yine de haberin kaynağı kontrol edilmelidir."
        )
    elif score < 60:
        return (
            "Girilen metin, sistem tarafından orta düzeyde dezenformasyon riski taşıyan "
            "bir içerik olarak değerlendirilmiştir. İçerikte bazı risk sinyalleri bulunmuştur; "
            "haber farklı kaynaklardan doğrulanmalıdır."
        )
    else:
        return (
            "Girilen metin, sistem tarafından yüksek dezenformasyon riski taşıyan "
            "bir içerik olarak değerlendirilmiştir. İçerik paylaşılmadan önce mutlaka "
            "güvenilir kaynaklardan doğrulanmalıdır."
        )
    """
    Nihai risk skoruna göre kısa açıklama üretir.
    """

    if score < 40:
        return (
            "Girilen metin, sistem tarafından düşük dezenformasyon riski taşıyan "
            "bir içerik olarak değerlendirilmiştir. Yine de haberin kaynağı kontrol edilmelidir."
        )

        def risk_explanation(score):
            """
            Nihai risk skoruna göre kısa açıklama üretir.
            """

            if score < 30:
                return (
                    "Girilen metin, sistem tarafından düşük dezenformasyon riski taşıyan "
                    "bir içerik olarak değerlendirilmiştir. Yine de haberin kaynağı kontrol edilmelidir."
                )
            elif score < 60:
                return (
                    "Girilen metin, sistem tarafından orta düzeyde dezenformasyon riski taşıyan "
                    "bir içerik olarak değerlendirilmiştir. İçerikte bazı risk sinyalleri bulunmuştur; "
                    "haber farklı kaynaklardan doğrulanmalıdır."
                )
            else:
                return (
                    "Girilen metin, sistem tarafından yüksek dezenformasyon riski taşıyan "
                    "bir içerik olarak değerlendirilmiştir. İçerik paylaşılmadan önce mutlaka "
                    "güvenilir kaynaklardan doğrulanmalıdır."
                )


@st.cache_resource
def load_model_and_vectorizer():
    """
    Kaydedilmiş modeli ve TF-IDF dönüştürücüyü yükler.
    """

    if not MODEL_PATH.exists():
        st.error(
            "Model dosyası bulunamadı. Önce 04_kalibre_svm_risk_modeli.py dosyasını çalıştırmalısın."
        )
        st.stop()

    if not VECTORIZER_PATH.exists():
        st.error(
            "Vectorizer dosyası bulunamadı. Önce 04_kalibre_svm_risk_modeli.py dosyasını çalıştırmalısın."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def calculate_ml_score(text, model, vectorizer):
    """
    Makine öğrenmesi modelinden sahte haber risk skorunu hesaplar.
    """

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_probability = probabilities[1]
    ml_score = round(fake_probability * 100, 2)

    if prediction == 1:
        prediction_label = "ML modeline göre sahte haber sınıfına yakın"
    else:
        prediction_label = "ML modeline göre gerçek haber sınıfına yakın"

    return prediction_label, ml_score


def calculate_criteria_scores(text):
    """
    Metni kriter bazlı analiz eder.
    Her kriter için 0-100 arası risk puanı üretir.
    """

    text_lower = text.lower()

    criteria_results = {}
    weighted_score = 0

    for criterion, keywords in CRITERIA_KEYWORDS.items():
        matched_keywords = []

        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)

        # Bir kriterde yakalanan her ifade risk puanını artırır.
        # 1 eşleşme = 35 puan
        # 2 eşleşme = 70 puan
        # 3 ve üzeri = 100 puan
        raw_score = min(len(matched_keywords) * 35, 100)

        weight = CRITERIA_WEIGHTS[criterion]
        weighted_score += raw_score * weight

        criteria_results[criterion] = {
            "score": raw_score,
            "weight": weight,
            "matched_keywords": matched_keywords,
        }
    # Kısa ve yüksek etkili iddialar için taban risk kuralı
    high_impact_keywords = [
        "öldü",
        "vefat etti",
        "hayatını kaybetti",
        "suikast",
        "darbe",
        "savaş başladı",
        "salgın başladı",
        "banka battı",
        "bankalar kapandı",
        "para çekme durduruldu",
        "hesaplar bloke edildi",
        "cumhurbaşkanı olacak",
        "cumhurbaskani olacak",
        "başkan olacak",
        "baskan olacak",
        "devlet başkanı olacak",
        "devlet baskani olacak",
        "seçimi kazanacak",
        "secimi kazanacak",
        "seçimi kazandı",
        "secimi kazandi",
        "aday oldu",
        "görevden alındı",
        "gorevden alindi",
        "istifa etti",
        "hükümet istifa edecek",
        "hukumet istifa edecek",
        "istifa edecek",
        "görevden ayrılacak",
        "gorevden ayrilacak",
        "hükümet düşecek",
        "hukumet dusecek",
        "erken seçim olacak",
        "erken secim olacak",
    ]

    uncertainty_keywords = [
        "iddialara göre",
        "iddialara gore",
        "iddia edildi",
        "iddia etti",
        "öne sürüldü",
        "one suruldu",
        "sosyal medyada yayıldı",
        "sosyal medyada yayildi",
        "doğrulanmadı",
        "dogrulanmadi",
        "teyit edilmedi",
        "tahminlere göre",
        "tahminlere gore",
        "iddialara göre",
        "iddialara gore",
        "öne sürüldü",
        "one suruldu",
        "iddia edildi",
        "iddia etti",
        "doğrulanmadı",
        "dogrulanmadi",
        "teyit edilmedi",
        "internet erişimi kapatılacak",
        "internet erisimi kapatilacak",
        "internet kapatılacak",
        "internet kapatilacak",
        "erişim kapatılacak",
        "erisim kapatilacak",
        "sosyal medya kapatılacak",
        "sosyal medya kapatilacak",
        "iletişim kesilecek",
        "iletisim kesilecek",
        "tüm şehirlerde internet",
        "tum sehirlerde internet",
    ]

    word_count = len(text.split())

    has_high_impact = any(keyword in text_lower for keyword in high_impact_keywords)
    has_uncertainty = any(keyword in text_lower for keyword in uncertainty_keywords)

    if word_count < 20 and has_high_impact:
        criteria_score = max(weighted_score, 45)

        criteria_results["Kaynak Eksikliği"]["score"] = max(
            criteria_results["Kaynak Eksikliği"]["score"], 70 if has_uncertainty else 35
        )

        criteria_results["Kaynak Eksikliği"]["matched_keywords"].append(
            "kısa metin + yüksek etkili doğrulanmamış iddia"
        )

        criteria_results["Kışkırtıcı Dil"]["score"] = max(
            criteria_results["Kışkırtıcı Dil"]["score"], 70
        )
        criteria_results["Kışkırtıcı Dil"]["matched_keywords"].append(
            "kısa metin + yüksek etkili iddia"
        )

        criteria_results["Duygu Aşırılığı"]["score"] = max(
            criteria_results["Duygu Aşırılığı"]["score"], 70
        )
        criteria_results["Duygu Aşırılığı"]["matched_keywords"].append(
            "kısa metin + yüksek etkili iddia"
        )

        weighted_score = 0
        for criterion, details in criteria_results.items():
            weighted_score += details["score"] * details["weight"]

        criteria_results["Duygu Aşırılığı"]["score"] = max(
            criteria_results["Duygu Aşırılığı"]["score"], 70
        )

        weighted_score = 0
        for criterion, details in criteria_results.items():
            weighted_score += details["score"] * details["weight"]
    criteria_score = round(weighted_score, 2)

    return criteria_score, criteria_results


def calculate_final_score(ml_score, criteria_score, text):
    """
    Makine öğrenmesi skoru ile kriter tabanlı skoru birleştirir.
    Kısa ve yüksek etkili iddialarda nihai skorun düşük riskte kalmasını engeller.
    """

    final_score = (0.30 * ml_score) + (0.70 * criteria_score)

    text_lower = text.lower()
    word_count = len(text.split())

    high_impact_keywords = [
        "öldü",
        "vefat etti",
        "hayatını kaybetti",
        "suikast",
        "darbe",
        "savaş başladı",
        "salgın başladı",
        "banka battı",
        "bankalar kapandı",
        "para çekme durduruldu",
        "hesaplar bloke edildi",
        "hükümet istifa edecek",
        "hukumet istifa edecek",
        "istifa edecek",
        "görevden ayrılacak",
        "gorevden ayrilacak",
        "hükümet düşecek",
        "hukumet dusecek",
        "erken seçim olacak",
        "erken secim olacak",
        "internet erişimi kapatılacak",
        "internet erisimi kapatilacak",
        "internet kapatılacak",
        "internet kapatilacak",
        "erişim kapatılacak",
        "erisim kapatilacak",
        "sosyal medya kapatılacak",
        "sosyal medya kapatilacak",
        "iletişim kesilecek",
        "iletisim kesilecek",
        "tüm şehirlerde internet",
        "tum sehirlerde internet",
    ]

    has_high_impact = any(keyword in text_lower for keyword in high_impact_keywords)

    if word_count < 20 and has_high_impact:
        final_score = max(final_score, 35)

    return round(final_score, 2)


def predict_news_risk(text, model, vectorizer):
    """
    Girilen haber metni için hibrit risk analizi yapar.
    """

    prediction_label, ml_score = calculate_ml_score(text, model, vectorizer)

    criteria_score, criteria_results = calculate_criteria_scores(text)

    final_score = calculate_final_score(ml_score, criteria_score, text)

    level = risk_level(final_score)
    explanation = risk_explanation(final_score)

    return {
        "prediction_label": prediction_label,
        "ml_score": ml_score,
        "criteria_score": criteria_score,
        "final_score": final_score,
        "risk_level": level,
        "explanation": explanation,
        "criteria_results": criteria_results,
    }


def main():
    st.set_page_config(
        page_title="Dezenformasyon Risk Analizi",
        page_icon="📰",
        layout="centered"
    )

    st.title("Türkçe Haber Metinlerinde Dezenformasyon Risk Analizi")

    st.write(
        "Bu uygulama, girilen haber metnini makine öğrenmesi modeli ve kriter tabanlı "
        "risk analiziyle değerlendirerek yüzdelik dezenformasyon risk skoru üretir."
    )

    st.warning(
        "Not: Bu sistem kesin doğrulama aracı değildir. Model, haber metninin dilsel "
        "örüntülerine ve belirlenen risk kriterlerine göre ön değerlendirme yapar. "
        "Daha güvenilir sonuçlar için tam haber metni girilmelidir."
    )

    model, vectorizer = load_model_and_vectorizer()

    st.subheader("Haber metni giriniz")

    user_text = st.text_area(
        label="Analiz edilecek haber metni",
        height=220,
        placeholder="Haber metnini buraya yapıştırınız..."
    )

    analyze_button = st.button("Analiz Et")

    if analyze_button:
        if user_text.strip() == "":
            st.error("Lütfen analiz etmek için bir haber metni giriniz.")
            return

        if len(user_text.split()) < 20:
            st.warning(
                "Girdiğiniz metin oldukça kısa. Model haber metinleriyle eğitildiği için "
                "daha güvenilir sonuç almak adına tam haber metni girmeniz önerilir."
            )

        result = predict_news_risk(user_text, model, vectorizer)

        st.subheader("Analiz Sonucu")

        st.metric("Nihai Dezenformasyon Risk Skoru", f"%{result['final_score']}")

        st.write("**Risk Seviyesi:**", result["risk_level"])
        if result["final_score"] >= 60:
            haber_degerlendirmesi = "Yüksek Riskli / Şüpheli Haber"
        elif result["final_score"] >= 30:
            haber_degerlendirmesi = "Orta Riskli / Doğrulanması Gereken Haber"
        else:
            haber_degerlendirmesi = "Düşük Riskli Haber"

        st.write("**Haber Değerlendirmesi:**", haber_degerlendirmesi)
        st.write("**Açıklama:**", result["explanation"])

        st.divider()

        st.subheader("Skor Bileşenleri")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Makine Öğrenmesi Skoru", f"%{result['ml_score']}")

        with col2:
            st.metric("Kriter Tabanlı Skor", f"%{result['criteria_score']}")

        st.info(
            "Kriter ağırlıkları, 83 katılımcıdan elde edilen anket yanıtlarının "
            "veri kalite kontrolü sonrası geçerli kabul edilen 67 yanıtı üzerinden hesaplanmıştır."
        )

        st.divider()

        st.subheader("Kriter Bazlı Analiz")

        for criterion, details in result["criteria_results"].items():
            st.write(f"**{criterion}**")
            st.write(f"Risk Puanı: %{details['score']}")
            st.write(f"Ağırlık: {details['weight']}")

            if details["matched_keywords"]:
                st.write("Eşleşen ifadeler:", ", ".join(details["matched_keywords"]))
            else:
                st.write("Eşleşen ifade bulunmadı.")

            st.write("---")

    st.caption(
        "Model: TF-IDF + Calibrated Linear SVM | "
        "Hibrit Skor: %30 Makine Öğrenmesi + %70 Kriter Tabanlı Analiz | " 
        "Etiketler: 0 = Gerçek Haber, 1 = Sahte Haber"
    )


if __name__ == "__main__":
    main()