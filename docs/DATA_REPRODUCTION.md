# Data sources and optional reproduction

The repository intentionally excludes raw news text corpora and individual survey answers. Rights to redistribute the external datasets have not been independently verified. If you have the right to use them, obtain the source data from the original publishers referenced in the academic thesis.

## Expected private source files

```text
data/
  fake.csv     # original file in user archive: Fake.csv (case difference)
  real.csv     # original file in user archive: Real.csv (case difference)
  ek_veri.xlsx # optional, supporting news dataset
  anket.ods    # optional, private raw survey responses; do NOT commit
```

**Important:** The original code in `src/01_veri_hazirlama.py` expects **lowercase** `fake.csv` and `real.csv`. Linux/macOS filenames are case-sensitive; name local copies accordingly. This repository preserves the original source instead of changing it.

For optional training and evaluation after placing permitted source files locally:

```bash
python src/01_veri_hazirlama.py
python src/02_model_egitimi.py
python src/04_kalibre_svm_risk_modeli.py
```

For optional survey processing, place the original `anket.ods` locally and run:

```bash
python src/05_anket_bwm_analizi.py
```

The separate `src/03_risk_skoru_test.py` is an earlier Logistic Regression experiment and not necessary to launch the final Streamlit app. It can be executed after data preparation if needed.

> **Caution:** Running model training may **overwrite** the saved `.pkl` model files or CSV result summaries with new artifacts. Preserve a backup and run it on a separate working copy if you want to retain identical reference results. Local source data and survey records should not be committed or shared publicly without proper authorization and privacy review.
