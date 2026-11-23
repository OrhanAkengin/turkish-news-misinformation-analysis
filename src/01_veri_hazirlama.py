from pathlib import Path
import pandas as pd


# Bu satır proje ana klasörünü otomatik bulur.
# Dosyamız src klasöründe olduğu için iki seviye yukarı çıkıyoruz.
BASE_DIR = Path(__file__).resolve().parent.parent

# data klasörünün yolu
DATA_DIR = BASE_DIR / "data"

# Girdi dosyaları
FAKE_PATH = DATA_DIR / "fake.csv"
REAL_PATH = DATA_DIR / "real.csv"
EXTRA_PATH = DATA_DIR / "ek_veri.xlsx"

# Çıktı dosyaları
OUTPUT_MAIN = DATA_DIR / "hazir_veri.csv"
OUTPUT_EXTRA = DATA_DIR / "ek_hazir_veri.csv"


def prepare_main_dataset():
    """
    Ana veri setini hazırlar.
    fake.csv ve real.csv dosyalarını okuyup tek veri seti haline getirir.
    """

    fake_df = pd.read_csv(FAKE_PATH)
    real_df = pd.read_csv(REAL_PATH)

    print("Fake.csv size:", fake_df.shape)
    print("Real.csv size:", real_df.shape)

    print("Fake.csv columns:", fake_df.columns.tolist())
    print("Real.csv columns:", real_df.columns.tolist())

    # Fake ve Real dosyalarında haber metinleri Body kolonunda.
    fake_df = fake_df[["Body"]].copy()
    real_df = real_df[["Body"]].copy()

    # Kolon adını standart hale getiriyoruz.
    fake_df.rename(columns={"Body": "text"}, inplace=True)
    real_df.rename(columns={"Body": "text"}, inplace=True)

    # Etiketleme:
    # 1 = sahte / yanlış haber
    # 0 = gerçek / doğru haber
    fake_df["label"] = 1
    real_df["label"] = 0

    # İki dosyayı alt alta birleştiriyoruz.
    df = pd.concat([fake_df, real_df], ignore_index=True)

    print("\nBefore cleaning:", df.shape)

    # Metinleri string yapıyoruz ve baş/son boşlukları temizliyoruz.
    df["text"] = df["text"].astype(str).str.strip()

    # Boş metinleri siliyoruz.
    df = df[df["text"] != ""]

    # Eğer boş değerler 'nan' yazısı olarak geldiyse onları da siliyoruz.
    df = df[df["text"].str.lower() != "nan"]

    # Aynı haber metni birden fazla kez varsa tekrarları siliyoruz.
    df = df.drop_duplicates(subset=["text"])

    # label kolonunu tam sayı yapıyoruz.
    df["label"] = df["label"].astype(int)

    print("After cleaning:", df.shape)

    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Temizlenmiş ana veri setini kaydediyoruz.
    df.to_csv(OUTPUT_MAIN, index=False, encoding="utf-8-sig")

    print("\nMain dataset saved:")
    print(OUTPUT_MAIN)


def prepare_extra_dataset():
    """
    Ek Excel veri setini hazırlar.
    Bu dosyada kolonlar:
    - Haberler
    - Sonuc

    Sonuc:
    Doğru / True  -> 0
    Yanlış / False -> 1
    """

    if not EXTRA_PATH.exists():
        print("\nExtra Excel file was not found. Skipping extra dataset.")
        return

    extra_df = pd.read_excel(EXTRA_PATH)

    print("\nExtra dataset size:", extra_df.shape)
    print("Extra dataset columns:", extra_df.columns.tolist())

    # Gerekli kolonları alıyoruz.
    extra_df = extra_df[["Haberler", "Sonuc"]].copy()

    # Haberler kolonunu text olarak yeniden adlandırıyoruz.
    extra_df.rename(columns={"Haberler": "text"}, inplace=True)

    # Sonuc kolonunu standart hale getiriyoruz.
    extra_df["Sonuc_clean"] = (
        extra_df["Sonuc"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Etiket dönüşümü
    label_map = {
        "doğru": 0,
        "dogru": 0,
        "true": 0,

        "yanlış": 1,
        "yanlis": 1,
        "false": 1
    }

    extra_df["label"] = extra_df["Sonuc_clean"].map(label_map)

    # Dönüştürülemeyen etiket var mı kontrol ediyoruz.
    missing = extra_df[extra_df["label"].isna()]

    if len(missing) > 0:
        print("\nWarning: Some labels could not be converted:")
        print(missing["Sonuc"].value_counts())

    # Etiketi çevrilemeyen satırları çıkarıyoruz.
    extra_df = extra_df.dropna(subset=["label"])

    # Sadece gerekli kolonları bırakıyoruz.
    extra_df = extra_df[["text", "label"]].copy()

    # Temizlik işlemleri
    extra_df["text"] = extra_df["text"].astype(str).str.strip()
    extra_df = extra_df[extra_df["text"] != ""]
    extra_df = extra_df[extra_df["text"].str.lower() != "nan"]
    extra_df = extra_df.drop_duplicates(subset=["text"])
    extra_df["label"] = extra_df["label"].astype(int)

    print("\nExtra dataset after cleaning:", extra_df.shape)

    print("\nExtra dataset class distribution:")
    print(extra_df["label"].value_counts())

    # Temizlenmiş ek veri setini kaydediyoruz.
    extra_df.to_csv(OUTPUT_EXTRA, index=False, encoding="utf-8-sig")

    print("\nExtra dataset saved:")
    print(OUTPUT_EXTRA)


def main():
    print("Data preparation started.\n")

    prepare_main_dataset()
    prepare_extra_dataset()

    print("\nData preparation completed.")


if __name__ == "__main__":
    main()