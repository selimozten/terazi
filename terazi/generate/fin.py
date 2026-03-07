"""Generator for terazi-fin: Turkish financial language tasks."""

from __future__ import annotations

from terazi.generate.base import BaseGenerator

SUBCATEGORIES = [
    "document_comprehension",
    "sentiment",
    "numerical_reasoning",
    "term_understanding",
]

SYSTEM_PROMPTS: dict[str, str] = {
    "document_comprehension": """Sen, Turkce finansal dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: BIST/KAP tarzi Turkce finansal dokuman anlama sorulari uretmek.

Kurallar:
- Turkiye finansal piyasalarina ozgu senaryolar: BIST sirketleri, KAP bildirimleri, faaliyet raporlari
- Gercekci ama kurgusal sirket adlari ve verileri kullan
- Sorular: onemli bilgiyi bulma, kararlarin etkisini anlama, finansal tablo okuma
- Turkce finansal terminoloji kullan (kar payi, sermaye artirimi, ozsermaye, FAVOK vb.)
- Zorluk: easy (dogrudan bilgi bulma), medium (cikarim), hard (coklu tablo analizi)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[KAP bildirimi veya finansal rapor metni]\n\nSoru: [soru]",
  "expected_output": "[dogru cevap]",
  "difficulty": "easy|medium|hard",
  "metadata": {"document_type": "...", "topic": "..."}
}""",
    "sentiment": """Sen, Turkce finansal duygu analizi degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce finansal haber ve yorumlardan duygu analizi ornekleri uretmek.

Kurallar:
- Turkiye ekonomisi ve BIST piyasasina dair gercekci haber basliklarI ve kisa metinler
- Duygu etiketleri: pozitif, negatif, notr
- Finansal jargon, piyasa terimleri ve ekonomik gostergeler icermeli
- Ince nüanslar iceren ornekler de olsun (mesela "beklenenden dusuk ama yine de kar")
- Zorluk: easy (acik duygu), medium (karisik sinyaller), hard (ince nuans)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "Asagidaki finansal metnin duygusunu belirleyin (pozitif/negatif/notr):\n\n[metin]",
  "expected_output": "[duygu etiketi ve kisa aciklama]",
  "difficulty": "easy|medium|hard",
  "metadata": {"sentiment": "pozitif|negatif|notr", "topic": "..."}
}""",
    "numerical_reasoning": """Sen, Turkce finansal sayisal muhakeme degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce finansal raporlardan sayisal muhakeme sorulari uretmek.

Kurallar:
- Gercekci finansal tablolar ve veriler icermeli (gelir tablosu, bilanco, nakit akisi)
- Hesaplama gerektiren sorular: yuzde degisim, oran analizi, karsilastirma
- Turkiye'ye ozgu finansal gostergeler: TUFE, politika faizi, doviz kurlari
- Sayilar tutarli ve mantikli olmali
- Zorluk: easy (tek hesaplama), medium (birden fazla adim), hard (karmasik analiz)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Finansal tablo veya veri seti]\n\nSoru: [sayisal muhakeme sorusu]",
  "expected_output": "[hesaplama adimları ve sonuc]",
  "difficulty": "easy|medium|hard",
  "metadata": {"calculation_type": "...", "topic": "..."}
}""",
    "term_understanding": """Sen, Turkce finansal terim degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce finansal terimlerin anlasilmasini olcen sorular uretmek.

Kurallar:
- BIST, SPK, BDDK, TCMB gibi kurumlara ait terimler
- Turkiye'ye ozgu finansal kavramlar: repo, ters repo, zorunlu karsiliklar, TL mevduat
- Terim tanimlamak, ornek vermek veya baglam icinde kullanmak seklinde sorular
- Zorluk: easy (yaygin terimler), medium (teknik terimler), hard (nadir/uzmanlik terimleri)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Finansal terim sorusu veya baglamsal kullanim]",
  "expected_output": "[dogru tanim/aciklama]",
  "difficulty": "easy|medium|hard",
  "metadata": {"term": "...", "domain": "..."}
}""",
}


class FinGenerator(BaseGenerator):
    category = "fin"

    def _get_subcategories(self) -> list[str]:
        return SUBCATEGORIES

    def _get_system_prompt(self, subcategory: str) -> str:
        return SYSTEM_PROMPTS[subcategory]

    def _get_user_prompt(self, subcategory: str, batch_size: int) -> str:
        return (
            f"{batch_size} adet yeni ve benzersiz Turkce finansal {subcategory} ornegi uret. "
            f"Zorluk dagilimi cesitli olsun. "
            f"Sadece JSON dizisi olarak cevap ver, baska bir sey yazma."
        )
