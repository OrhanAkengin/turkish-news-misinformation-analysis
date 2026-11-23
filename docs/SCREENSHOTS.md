# Screenshot index and provenance

All screenshots below were copied **unchanged at the pixel level** from the user-provided image attachments; only their filenames were standardized in this public portfolio folder. Each numbered `-1`, `-2`, `-3` series represents successive portions of **one** original Streamlit analysis view, not three independent experiments.

| English repo filename | Original attachment filename | Content |
|---|---|---|
| `application-home.png` | `Ana ekran.png` | Home screen |
| `health-claim-analysis-1.png` | `Sağlık İddiası İçeren Orta Riskli Haber Analizi-1(1).png` | Health claim: final score / scoring components / criterion details |
| `health-claim-analysis-2.png` | `Sağlık İddiası İçeren Orta Riskli Haber Analizi-2(1).png` | Health claim: final score / scoring components / criterion details |
| `health-claim-analysis-3.png` | `Sağlık İddiası İçeren Orta Riskli Haber Analizi-3(1).png` | Health claim: final score / scoring components / criterion details |
| `low-risk-analysis-1.png` | `Düşük risk-1(2).png` | Low risk: result / scoring components / criterion details |
| `low-risk-analysis-2.png` | `Düşük risk-2(1).png` | Low risk: result / scoring components / criterion details |
| `low-risk-analysis-3.png` | `Düşük risk-3(1).png` | Low risk: result / scoring components / criterion details |
| `misleading-context-analysis-1.png` | `Hatalı Bağlam İçeren Haber Analizi-1(1).png` | Misleading context: result / scoring components / criterion details |
| `misleading-context-analysis-2.png` | `Hatalı Bağlam İçeren Haber Analizi-2(1).png` | Misleading context: result / scoring components / criterion details |
| `misleading-context-analysis-3.png` | `Hatalı Bağlam İçeren Haber Analizi-3(1).png` | Misleading context: result / scoring components / criterion details |
| `short-high-impact-claim-1.png` | `Kısa ve Yüksek Etkili İddia Analizi-1(1).png` | Short high-impact claim: result / scoring components / criterion details |
| `short-high-impact-claim-2.png` | `Kısa ve Yüksek Etkili İddia Analizi-2(1).png` | Short high-impact claim: result / scoring components / criterion details |
| `short-high-impact-claim-3.png` | `Kısa ve Yüksek Etkili İddia Analizi-3(1).png` | Short high-impact claim: result / scoring components / criterion details |

## Reproduced reference outputs

| Case | ML component | Criteria component | Final risk | Risk level |
|---|---:|---:|---:|---|
| Meteorological bulletin | 0.78% | 0.00% | 0.23% | Low |
| Unsupported health claim | 7.96% | 56.46% | 41.91% | Medium |
| Misleading-context claim | 2.12% | 56.34% | 40.07% | Medium |
| Short high-impact claim | 27.89% | 0.00% | 35.00% | Medium |

The short-claim example is affected by the application's special minimum-final-risk rule. Screenshots retain Turkish user-interface text because they document the original application as built. See `docs/VALIDATION.md`.
