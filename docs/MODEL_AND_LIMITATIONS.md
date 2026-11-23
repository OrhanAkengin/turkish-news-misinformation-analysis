# Model card and scope

## Intended use

A student-built decision-support **pre-screening** demonstration for linguistic features associated with misinformation risk in **Turkish news text**. It is **not** a fact-checking tool and should never be used as the sole basis to label a real article false or true.

## Training and evaluation basis

- Main data: `Fake.csv` (2,162 fake/satirical items) and `Real.csv` (2,293 real items) before cleanup; **4,451** items after four duplicates were removed.
- Main data after cleanup: **2,160** fake/satirical and **2,291** real items.
- Training and test: **3,560** / **891** texts, with 10,000 TF-IDF features using unigram/bigram representation.
- Calibrated Linear SVM test-set metrics as saved in `outputs/calibrated_svm_sonuclari.csv`: accuracy **0.9876543**, precision **0.9883991**, recall **0.9861111**, F1 **0.9872538**.
- Linear SVM, Logistic Regression, Naive Bayes and Random Forest were also compared; the calibrated SVM was used in the final app. Results are dataset-specific, not evidence of external cross-source validation.
- The additional cleaned dataset (3,513 examples) was prepared separately; it was not used for the reported final model test scores.

## Risk evaluation

The model's calibrated fake-class score is combined with four text indicator criteria:

- Missing source / citation (`Kaynak Eksikliği`), weight ~0.332090
- Misleading context (`Hatalı Bağlam`), weight ~0.231343
- Emotional exaggeration (`Duygu Aşırılığı`), weight ~0.227612
- Inflammatory language (`Kışkırtıcı Dil`), weight ~0.208955

Weights came from **67 valid answers out of 83 survey responses** through a **BWM-inspired best/worst choice normalization**. This is not full standard BWM optimization and does not represent expert consensus.

The base final score is `0.30 × ML score + 0.70 × criteria score`. Risk bands are <30 low, 30–<60 medium, and ≥60 high. Certain short, high-impact statements receive at least 35% final risk as an additional heuristic.

## Technical limitations / implementation notes

1. **Overlapping and repeated phrases:** `src/app.py` searches by substring, and duplicate or nested expressions can be counted more than once. This may inflate the criteria score. It is kept unchanged here to preserve original results.
2. **UI snapshot differences:** user-provided example screenshots show a model label different from the label in the current `app.py` source. Calculated sample outputs match, but the exact presented UI revision may differ.
3. **Different experimental thresholds:** auxiliary scripts such as `03_risk_skoru_test.py` and `04_kalibre_svm_risk_modeli.py` include 40/70 example thresholds, while `app.py` and the thesis use 30/60. The deployed app source takes precedence for its own output.
4. **Limited contextual understanding:** simple keyword rules do not resolve irony, context, uncertainty, or whether a claim is actually true.
5. **Dataset shift:** the training corpus may not represent current reporting styles or newly emerging misinformation patterns.
6. **Calibration ≠ fact-checking:** the numerical score should not be interpreted as the probability that a given story is false in real-world use.
7. **Pickle security and versioning:** never unpickle model artifacts from unknown sources; saved artifacts may require compatible scikit-learn versions.

Code and original application behavior have **not** been deliberately refactored in this GitHub-ready package.
