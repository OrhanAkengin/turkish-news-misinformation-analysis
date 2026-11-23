from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

ANKET_PATH = DATA_DIR / "anket.ods"

OUTPUT_WEIGHTS_PATH = OUTPUTS_DIR / "anket_kriter_agirliklari_duzeltilmis.csv"
OUTPUT_VALID_PATH = OUTPUTS_DIR / "anket_gecerli_cevaplar_duzeltilmis.csv"
OUTPUT_INVALID_PATH = OUTPUTS_DIR / "anket_gecersiz_cevaplar_duzeltilmis.csv"
OUTPUT_QUALITY_SUMMARY_PATH = OUTPUTS_DIR / "anket_veri_kalite_ozeti_duzeltilmis.csv"


CRITERIA = [
    "Kaynak Eksikliği",
    "Kışkırtıcı Dil",
    "Duygu Aşırılığı",
    "Hatalı Bağlam"
]


def normalize_text(value):
    """
    Türkçe karakter, boşluk ve büyük/küçük harf farklarını azaltır.
    """

    value = str(value).strip().lower()

    replacements = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
        "İ": "i",
    }

    for tr_char, en_char in replacements.items():
        value = value.replace(tr_char, en_char)

    value = re.sub(r"\s+", " ", value)

    return value


def match_criterion(value):
    """
    Anket cevabındaki metni dört ana kriterden biriyle eşleştirir.
    """

    value_norm = normalize_text(value)

    criterion_patterns = {
        "Kaynak Eksikliği": [
            "kaynak",
            "kaynak eksikligi",
            "resmi kaynak"
        ],
        "Kışkırtıcı Dil": [
            "kiskirtici",
            "kiskirtici dil",
            "provokatif"
        ],
        "Duygu Aşırılığı": [
            "duygu",
            "duygu asiriligi",
            "duygusal"
        ],
        "Hatalı Bağlam": [
            "hatali baglam",
            "baglam"
        ],
    }

    for criterion, patterns in criterion_patterns.items():
        for pattern in patterns:
            if normalize_text(pattern) in value_norm:
                return criterion

    return None


def to_number(value):
    """
    Anket puanlarını güvenli şekilde sayıya dönüştürür.
    1-9 arası olmayan değerlerde None döndürür.
    """

    if pd.isna(value):
        return None

    value_str = str(value).strip().replace(",", ".")

    try:
        number = float(value_str)
    except ValueError:
        return None

    if number < 1 or number > 9:
        return None

    return number


def find_comparison_columns(df, question_number):
    """
    Soru 3 veya Soru 4'e ait kriter puanlama kolonlarını bulur.

    question_number = "3)" veya "4)"
    """

    columns = {}

    for col in df.columns:
        col_norm = normalize_text(col)

        if question_number not in col_norm:
            continue

        for criterion in CRITERIA:
            criterion_norm = normalize_text(criterion)
            if criterion_norm in col_norm:
                columns[criterion] = col

    return columns


def load_survey():
    """
    Anket dosyasını okur.
    """

    if not ANKET_PATH.exists():
        raise FileNotFoundError(
            f"{ANKET_PATH} bulunamadı. Lütfen anket dosyasını data/anket.ods olarak kaydet."
        )

    df = pd.read_excel(ANKET_PATH, engine="odf")

    print("Survey dataset size:", df.shape)

    print("\nSurvey columns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    return df


def validate_and_repair_responses(df):
    """
    Anket cevaplarını kalite kontrolünden geçirir.

    Bu sürümde düzeltilebilir hatalar silinmez:
    - En önemli kriterin kendi karşılaştırma puanı 1 değilse 1 yapılır.
    - En önemsiz kriterin kendi karşılaştırma puanı 1 değilse 1 yapılır.

    Ancak şu durumlar geçersiz sayılır:
    - En önemli kriter eşleştirilemezse
    - En önemsiz kriter eşleştirilemezse
    - En önemli ve en önemsiz kriter aynı seçilmişse
    - Puanlardan biri boşsa veya 1-9 aralığında değilse
    """

    best_col = df.columns[1]
    worst_col = df.columns[2]

    best_to_others_cols = find_comparison_columns(df, "3)")
    others_to_worst_cols = find_comparison_columns(df, "4)")

    print("\nDetected best criterion column:")
    print(best_col)

    print("\nDetected worst criterion column:")
    print(worst_col)

    print("\nDetected Soru 3 comparison columns:")
    print(best_to_others_cols)

    print("\nDetected Soru 4 comparison columns:")
    print(others_to_worst_cols)

    if len(best_to_others_cols) != 4:
        raise ValueError("Soru 3 için dört kriter kolonu bulunamadı.")

    if len(others_to_worst_cols) != 4:
        raise ValueError("Soru 4 için dört kriter kolonu bulunamadı.")

    valid_rows = []
    invalid_rows = []

    total_repairs = 0

    for index, row in df.iterrows():
        reasons = []
        repairs = []

        best_criterion = match_criterion(row[best_col])
        worst_criterion = match_criterion(row[worst_col])

        if best_criterion is None:
            reasons.append("En önemli kriter eşleştirilemedi.")

        if worst_criterion is None:
            reasons.append("En önemsiz kriter eşleştirilemedi.")

        if best_criterion is not None and worst_criterion is not None:
            if best_criterion == worst_criterion:
                reasons.append("En önemli ve en önemsiz kriter aynı seçilmiş.")

        best_to_others_scores = {}
        others_to_worst_scores = {}

        for criterion in CRITERIA:
            score_3 = to_number(row[best_to_others_cols[criterion]])
            score_4 = to_number(row[others_to_worst_cols[criterion]])

            best_to_others_scores[criterion] = score_3
            others_to_worst_scores[criterion] = score_4

            if score_3 is None:
                reasons.append(f"Soru 3 - {criterion} puanı boş veya 1-9 aralığında değil.")

            if score_4 is None:
                reasons.append(f"Soru 4 - {criterion} puanı boş veya 1-9 aralığında değil.")

        # Düzeltilebilir hata 1:
        # En önemli kriterin kendisiyle karşılaştırma puanı tanımsal olarak 1 olmalıdır.
        if best_criterion is not None and best_criterion in best_to_others_scores:
            self_score_best = best_to_others_scores[best_criterion]

            if self_score_best is not None and self_score_best != 1:
                best_to_others_scores[best_criterion] = 1
                repairs.append(
                    f"Soru 3 öz-karşılaştırma düzeltildi: {best_criterion} {self_score_best} -> 1"
                )
                total_repairs += 1

        # Düzeltilebilir hata 2:
        # En önemsiz kriterin kendisiyle karşılaştırma puanı tanımsal olarak 1 olmalıdır.
        if worst_criterion is not None and worst_criterion in others_to_worst_scores:
            self_score_worst = others_to_worst_scores[worst_criterion]

            if self_score_worst is not None and self_score_worst != 1:
                others_to_worst_scores[worst_criterion] = 1
                repairs.append(
                    f"Soru 4 öz-karşılaştırma düzeltildi: {worst_criterion} {self_score_worst} -> 1"
                )
                total_repairs += 1

        output_row = row.to_dict()
        output_row["best_criterion_clean"] = best_criterion
        output_row["worst_criterion_clean"] = worst_criterion
        output_row["validation_status"] = "valid" if len(reasons) == 0 else "invalid"
        output_row["invalid_reasons"] = " | ".join(reasons)
        output_row["repair_notes"] = " | ".join(repairs)

        for criterion in CRITERIA:
            output_row[f"Soru3_{criterion}"] = best_to_others_scores[criterion]
            output_row[f"Soru4_{criterion}"] = others_to_worst_scores[criterion]

        if len(reasons) == 0:
            valid_rows.append(output_row)
        else:
            invalid_rows.append(output_row)

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    return valid_df, invalid_df, total_repairs


def calculate_weights(valid_df):
    """
    Geçerli cevaplar üzerinden kriter ağırlıklarını hesaplar.

    Temel yaklaşım:
    - En önemli seçilme oranı ağırlığı artırır.
    - En önemsiz seçilme oranı ağırlığı azaltır.
    """

    if len(valid_df) == 0:
        raise ValueError("Geçerli anket cevabı yok. Ağırlık hesaplanamaz.")

    best_counts = valid_df["best_criterion_clean"].value_counts()
    worst_counts = valid_df["worst_criterion_clean"].value_counts()

    total_valid = len(valid_df)

    rows = []

    for criterion in CRITERIA:
        best_count = int(best_counts.get(criterion, 0))
        worst_count = int(worst_counts.get(criterion, 0))

        best_ratio = best_count / total_valid
        worst_ratio = worst_count / total_valid

        raw_score = best_ratio + (1 - worst_ratio)

        rows.append({
            "criterion": criterion,
            "best_count": best_count,
            "worst_count": worst_count,
            "best_ratio": best_ratio,
            "worst_ratio": worst_ratio,
            "raw_score": raw_score
        })

    weights_df = pd.DataFrame(rows)

    total_raw = weights_df["raw_score"].sum()
    weights_df["weight"] = weights_df["raw_score"] / total_raw

    weights_df = weights_df.sort_values(by="weight", ascending=False)

    return weights_df


def save_outputs(valid_df, invalid_df, weights_df, original_count, total_repairs):
    """
    Analiz çıktılarının tamamını kaydeder.
    """

    OUTPUTS_DIR.mkdir(exist_ok=True)

    valid_df.to_csv(OUTPUT_VALID_PATH, index=False, encoding="utf-8-sig")
    invalid_df.to_csv(OUTPUT_INVALID_PATH, index=False, encoding="utf-8-sig")
    weights_df.to_csv(OUTPUT_WEIGHTS_PATH, index=False, encoding="utf-8-sig")

    quality_summary = pd.DataFrame({
        "metric": [
            "Toplam yanıt sayısı",
            "Geçerli yanıt sayısı",
            "Geçersiz yanıt sayısı",
            "Düzeltilen öz-karşılaştırma sayısı",
            "Geçerli yanıt oranı",
            "Geçersiz yanıt oranı",
            "En yüksek ağırlıklı kriter",
            "En düşük ağırlıklı kriter"
        ],
        "value": [
            original_count,
            len(valid_df),
            len(invalid_df),
            total_repairs,
            round(len(valid_df) / original_count, 4) if original_count > 0 else 0,
            round(len(invalid_df) / original_count, 4) if original_count > 0 else 0,
            weights_df.iloc[0]["criterion"],
            weights_df.iloc[-1]["criterion"]
        ]
    })

    quality_summary.to_csv(
        OUTPUT_QUALITY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nData quality summary:")
    print(quality_summary)

    print("\nCalculated repaired criterion weights:")
    print(weights_df)

    print("\nValid repaired responses saved:")
    print(OUTPUT_VALID_PATH)

    print("\nInvalid repaired responses saved:")
    print(OUTPUT_INVALID_PATH)

    print("\nRepaired weights saved:")
    print(OUTPUT_WEIGHTS_PATH)

    print("\nQuality summary saved:")
    print(OUTPUT_QUALITY_SUMMARY_PATH)


def main():
    print("Survey data repair and quality analysis started.\n")

    df = load_survey()

    valid_df, invalid_df, total_repairs = validate_and_repair_responses(df)

    print("\nTotal responses:", len(df))
    print("Valid responses after repair:", len(valid_df))
    print("Invalid responses after repair:", len(invalid_df))
    print("Total self-comparison repairs:", total_repairs)

    weights_df = calculate_weights(valid_df)

    save_outputs(
        valid_df=valid_df,
        invalid_df=invalid_df,
        weights_df=weights_df,
        original_count=len(df),
        total_repairs=total_repairs
    )

    print("\nSurvey data repair and quality analysis completed.")


if __name__ == "__main__":
    main()