"""Generator for terazi-legal: Turkish legal language tasks."""

from __future__ import annotations

from terazi.generate.base import BaseGenerator

SUBCATEGORIES = [
    "document_comprehension",
    "case_reasoning",
    "clause_extraction",
    "regulatory_compliance",
]

SYSTEM_PROMPTS: dict[str, str] = {
    "document_comprehension": """Sen, Turkce hukuki dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce hukuki dokuman anlama sorulari uretmek.

Kurallar:
- Turkiye hukuk sistemine uygun dokümanlar: kanun maddeleri, yonetmelikler, tebligler
- Turk Borçlar Kanunu, Turk Ticaret Kanunu, Is Kanunu, Ceza Kanunu gibi temel mevzuat
- Sorular: madde yorumlama, uygulanabilirlik, kosul belirleme
- Resmi hukuki Turkce kullan
- Zorluk: easy (acik madde bulma), medium (yorum gerektiren), hard (coklu mevzuat karsilastirma)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Hukuki metin]\n\nSoru: [soru]",
  "expected_output": "[dogru cevap ve hukuki dayanak]",
  "difficulty": "easy|medium|hard",
  "metadata": {"law": "...", "topic": "..."}
}""",
    "case_reasoning": """Sen, Turkce hukuki muhakeme degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce ictihat muhakemesi ornekleri uretmek.

Kurallar:
- Yargitay/Danistay karar ozeti tarzinda ornekler
- Gercekci ama kurgusal dava senaryolari
- Hukuki muhakeme gerektiren sorular: emsal uygulama, hak belirleme, sorumluluk tespiti
- Turkiye hukuk sistemine ozgu kavramlar: kusur, tazminat, hukmun bozulmasi, temyiz
- Zorluk: medium (tek hukuki mesele), hard (coklu mesele, catisan haklar)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Dava ozeti ve olgular]\n\nSoru: [hukuki muhakeme sorusu]",
  "expected_output": "[beklenen hukuki analiz]",
  "difficulty": "medium|hard",
  "metadata": {"area": "...", "court": "..."}
}""",
    "clause_extraction": """Sen, Turkce sozlesme analizi degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce sozlesmelerden madde cikarimi ve analizi ornekleri uretmek.

Kurallar:
- Gercekci Turkce sozlesme metinleri: is sozlesmesi, kira sozlesmesi, ticari sozlesme
- Gorevler: belirli bir maddeyi bulma, kosullari belirleme, taraf yukumluluklerini cikarma
- Turkce hukuki sozlesme dili kullan
- Zorluk: easy (acik madde bulma), medium (kosul analizi), hard (catisan maddeler)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Sozlesme metni]\n\nGorev: [cikarim gorevi]",
  "expected_output": "[cikarilan bilgi]",
  "difficulty": "easy|medium|hard",
  "metadata": {"contract_type": "...", "task_type": "..."}
}""",
    "regulatory_compliance": """Sen, Turkce mevzuat uyumu degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce mevzuat uyum sorulari uretmek.

Kurallar:
- Turkiye'deki duzenleyici kurumlar ve mevzuat: SPK, BDDK, KVKK, Rekabet Kurumu, SGK
- Senaryo bazli sorular: "Bu durumda hangi mevzuata uyulmali? Ne yapilmali?"
- KVKK (kisisel verilerin korunmasi), is hukuku, vergi mevzuati gibi guncel konular
- Zorluk: easy (tek mevzuat, acik kural), medium (birden fazla mevzuat), hard (gri alan, yorum)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Senaryo tanimi]\n\nSoru: [mevzuat uyumu sorusu]",
  "expected_output": "[uygulanacak mevzuat ve gerekli adimlar]",
  "difficulty": "easy|medium|hard",
  "metadata": {"regulation": "...", "sector": "..."}
}""",
}


class LegalGenerator(BaseGenerator):
    category = "legal"

    def _get_subcategories(self) -> list[str]:
        return SUBCATEGORIES

    def _get_system_prompt(self, subcategory: str) -> str:
        return SYSTEM_PROMPTS[subcategory]

    def _get_user_prompt(self, subcategory: str, batch_size: int) -> str:
        return (
            f"{batch_size} adet yeni ve benzersiz Turkce hukuki {subcategory} ornegi uret. "
            f"Zorluk dagilimi cesitli olsun. "
            f"Sadece JSON dizisi olarak cevap ver, baska bir sey yazma."
        )
