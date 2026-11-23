# Validation and source integrity

This package was prepared from `tez.zip` (original project) and the user's 13 PNG captures. The original six Python source files and the two inference model files were copied **byte-for-byte**, without refactoring, patching, renaming the internal source files, or altering score calculations. SHA-256 checksums for those files are in `SOURCE_SHA256.txt`.

## Confirmed tests (before packaging)

- Six original `.py` modules passed Python syntax parsing.
- The saved calibrated model and TF-IDF vectorizer loaded with scikit-learn 1.8.0 in the inspection environment.
- Recomputed sample scores from four user-provided text examples matched the screenshot reference values: **0.23%, 41.91%, 40.07%, 35.00%**.
- Reported calibrated-model evaluation CSV matches the capstone document's metrics.
- Dataset preparation and survey aggregation were separately checked in the earlier analysis; the source datasets and survey rows are not redistributed here.

## Not verified

- Interactive Streamlit browser behavior on a fresh user's computer.
- Fresh installation of all dependencies on every OS/Python combination.
- General accuracy on live or independently verified external news.
- Pixel-perfect reproduction of the screenshots from the packaged `app.py` (the UI revision differs).

## Repeat simple regression checks

After installing dependencies:

```bash
python -m unittest discover -s tests -v
```

This regression test loads only the two packaged inference model artifacts and computes the four example scores using unchanged app functions. It does not recreate the whole train/test split.
