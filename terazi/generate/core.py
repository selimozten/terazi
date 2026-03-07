"""Generator for terazi-core: general Turkish language understanding."""

from __future__ import annotations

from terazi.generate.base import BaseGenerator

SUBCATEGORIES = [
    "reading_comprehension",
    "common_sense",
    "grammar",
    "translation",
    "summarization",
]

SYSTEM_PROMPTS: dict[str, str] = {
    "reading_comprehension": """Sen, Turkce dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce okuduğunu anlama sorulari uretmek.

Kurallar:
- Her soru bir Turkce metin paragrafı ve bu paragrafa dayanan bir soru icermelidir
- Metinler kulturel olarak Turkiye'ye ozgu olmali (tarih, cografya, gunluk yasam, edebiyat)
- Sorular acik uclu degil, net bir cevabi olan sorular olmali
- Zorluk seviyeleri: easy (basit bilgi bulma), medium (cikarim gerektiren), hard (coklu cikarim + analiz)
- Metinler dogal Turkce olmali, Ingilizce'den ceviri hissi vermemeli
- Dil bilgisi ve imla kurallarina uygun olmali

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "Metin: [paragraf]\n\nSoru: [soru]",
  "expected_output": "[dogru cevap]",
  "difficulty": "easy|medium|hard",
  "metadata": {"topic": "..."}
}""",
    "common_sense": """Sen, Turkce dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce sagduyu muhakemesi sorulari uretmek.

Kurallar:
- Sorular Turkiye'deki gunluk yasam, kultur ve genel bilgiye dayali olmali
- Turkiye'ye ozgu senaryolar kullan (bayramlar, yemek kulturu, sehirler, toplumsal normlar)
- Cevaplar mantiksal cikarimla ulasilabilir olmali
- Her soru icin 4 secenekli coktan secmeli format kullan
- Zorluk seviyeleri: easy (acik sagduyu), medium (biraz dusunme gerektiren), hard (derin kulturel bilgi)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Soru]\nA) ...\nB) ...\nC) ...\nD) ...",
  "expected_output": "[dogru secenek harfi ve aciklama]",
  "difficulty": "easy|medium|hard",
  "metadata": {"topic": "..."}
}""",
    "grammar": """Sen, Turkce dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce dil bilgisi ve dilbilim sorulari uretmek.

Kurallar:
- Turkce'ye ozgu dil bilgisi konularini kapsa: ekler, fiil cekimleri, sozcuk turleri, cumle yapisi
- Unlu uyumu, unsuz yumusamasi, hal ekleri gibi Turkce'ye has konular icermeli
- Turkce'nin sondan eklemeli yapisini test eden sorular olsun
- Zorluk seviyeleri: easy (temel ek bilgisi), medium (karmasik cumle analizi), hard (dilbilimsel aciklama)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "[Dil bilgisi sorusu veya cumle analizi gorevi]",
  "expected_output": "[dogru cevap ve aciklama]",
  "difficulty": "easy|medium|hard",
  "metadata": {"topic": "..."}
}""",
    "translation": """Sen, Turkce dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce-Ingilizce ceviri kalitesi degerlendirme ornekleri uretmek.

Kurallar:
- Hem TR->EN hem EN->TR yonlerinde ornekler uret
- Deyimler, atasozleri ve kulturel ifadeler icermeli
- Teknik, edebi ve gunluk dil cesitliligi olmali
- Bire bir ceviri yerine anlam aktarimini olcen ornekler
- Zorluk: easy (basit cumleler), medium (deyimsel ifadeler), hard (edebi/teknik metin)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "Cevirin: [kaynak metin]\nYon: [TR->EN veya EN->TR]",
  "expected_output": "[referans ceviri]",
  "difficulty": "easy|medium|hard",
  "metadata": {"direction": "TR->EN|EN->TR", "domain": "..."}
}""",
    "summarization": """Sen, Turkce dil degerlendirme veri seti olusturan bir uzmansin.

Gorevin: Turkce metin ozetleme gorevleri uretmek.

Kurallar:
- Cesitli turde Turkce metinler uret: haber, makale, hikaye, konusma
- Metinler 150-400 kelime arasinda olmali
- Beklenen ozet 2-4 cumle olmali
- Turkiye gundemine, kulturune ve toplumsal konulara dair metinler
- Zorluk: easy (acik ana fikir), medium (birden fazla onemli nokta), hard (karmasik argumanlar)

Cikti formati: JSON dizisi. Her eleman:
{
  "input": "Asagidaki metni ozetleyin:\n\n[metin]",
  "expected_output": "[referans ozet]",
  "difficulty": "easy|medium|hard",
  "metadata": {"domain": "...", "word_count": N}
}""",
}


class CoreGenerator(BaseGenerator):
    category = "core"

    def _get_subcategories(self) -> list[str]:
        return SUBCATEGORIES

    def _get_system_prompt(self, subcategory: str) -> str:
        return SYSTEM_PROMPTS[subcategory]

    def _get_user_prompt(self, subcategory: str, batch_size: int) -> str:
        difficulty_note = (
            "Zorluk dagilimi: yaklasik 1/3 easy, 1/3 medium, 1/3 hard olsun."
        )
        return (
            f"{batch_size} adet yeni ve benzersiz Turkce {subcategory} ornegi uret. "
            f"{difficulty_note} "
            f"Sadece JSON dizisi olarak cevap ver, baska bir sey yazma."
        )
