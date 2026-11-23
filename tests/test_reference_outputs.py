"""Smoke tests for four outputs preserved in original Streamlit screenshots.

Run from project root after installing requirements:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app  # Original app.py; intentionally unchanged.


SAMPLES = [
    (
        "official_meteorology",
        "Meteoroloji Genel Müdürlüğü, hafta sonu bazı bölgelerde kuvvetli sağanak yağış beklendiğini açıkladı. Yapılan duyuruda, sel, su baskını ve ulaşımda aksamalara karşı vatandaşların dikkatli ve tedbirli olması gerektiği belirtildi. Açıklamada, uyarıların resmi meteorolojik değerlendirmelere dayandığı ve gelişmelerin kurumun duyuru kanalları üzerinden takip edilmesi gerektiği ifade edildi.",
        0.23, "Düşük Risk",
    ),
    (
        "unsupported_health_claim",
        "Son dakika! Sosyal medyada yayılan iddiaya göre, evde hazırlanan özel bir karışımın kanser, diyabet ve kalp hastalıklarını tamamen iyileştirdiği öne sürüldü. Paylaşımda kullanıcıların doktora gitmeden bu yöntemi hemen denemesi gerektiği belirtildi. İçerikte herhangi bir bilimsel araştırmaya, uzman görüşüne veya resmi sağlık kurumu açıklamasına yer verilmediği halde, bu yöntemin ilaç şirketleri tarafından halktan saklandığı iddia edildi.",
        41.91, "Orta Risk",
    ),
    (
        "misleading_context",
        "Sosyal medyada paylaşılan görüntülerin bugün yaşanan büyük bir protestoya ait olduğu iddia edildi. Ancak videonun eski bir görüntü olduğu ve farklı bir ülkede gerçekleşen başka bir olaydan alındığı öne sürüldü. Paylaşımda görüntülerin tarihine, kaynağına ve olayın bağlamına ilişkin doğrulanabilir bilgiye yer verilmedi. Kullanıcıların videoyu gerçekmiş gibi hızla paylaşması istendi.",
        40.07, "Orta Risk",
    ),
    (
        "short_high_impact",
        "İddialara göre yarın sabah tüm şehirlerde internet erişimi kapatılacak.",
        35.00, "Orta Risk",
    ),
]


class OriginalScreenshotRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vectorizer = app.load_model_and_vectorizer()

    def test_reference_outputs(self):
        for name, article, expected_score, expected_level in SAMPLES:
            with self.subTest(case=name):
                actual = app.predict_news_risk(article, self.model, self.vectorizer)
                self.assertAlmostEqual(actual["final_score"], expected_score, places=2)
                self.assertEqual(actual["risk_level"], expected_level)


if __name__ == "__main__":
    unittest.main()
