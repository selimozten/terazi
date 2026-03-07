"""Generator for terazi-tool: tool use and function calling in Turkish."""

from __future__ import annotations

from terazi.generate.base import BaseGenerator

SUBCATEGORIES = [
    "api_call",
    "multi_step",
    "parameter_extraction",
    "error_recovery",
]

MOCK_TOOLS = """
Kullanilabilir araclar (tools):

1. hava_durumu(sehir: str, tarih: str | None) -> {"sicaklik": float, "durum": str, "nem": float}
   Belirtilen sehir icin hava durumu bilgisi dondurur.

2. restoran_ara(konum: str, mutfak: str | None, min_puan: float | None) -> [{"ad": str, "puan": float, "adres": str}]
   Restoranları arar ve listeler.

3. ucus_ara(kalkis: str, varis: str, tarih: str, yolcu_sayisi: int) -> [{"sefer": str, "saat": str, "fiyat": float}]
   Ucak bileti arar.

4. doviz_kuru(kaynak: str, hedef: str, miktar: float) -> {"sonuc": float, "kur": float}
   Doviz cevirisi yapar.

5. haber_getir(kategori: str, adet: int) -> [{"baslik": str, "ozet": str, "tarih": str}]
   Belirtilen kategorideki haberleri getirir.

6. takvim_olustur(baslik: str, tarih: str, saat: str, sure_dk: int, katilimcilar: list[str] | None) -> {"id": str, "durum": str}
   Takvimde yeni bir etkinlik olusturur.

7. mesaj_gonder(alici: str, icerik: str, oncelik: str | None) -> {"durum": str, "id": str}
   Mesaj gonderir.

8. dosya_ara(sorgu: str, dosya_tipi: str | None, klasor: str | None) -> [{"ad": str, "yol": str, "boyut": int}]
   Dosya arar.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "api_call": f"""Sen, Turkce arac kullanimi (tool use) degerlendirme veri seti olusturan bir uzmansin.

{MOCK_TOOLS}

Gorevin: Turkce dogal dil talimatlarindan fonksiyon cagrisi olusturma ornekleri uretmek.

Kurallar:
- Kullanici Turkce bir talep yazar, model dogru araci dogru parametrelerle cagirmali
- Talepler dogal ve cesitli olmali (resmi, gunluk, belirsiz)
- Beklenen cikti: arac adi ve parametreleri iceren JSON
- Zorluk: easy (acik tek arac), medium (belirsiz parametreler), hard (birden fazla olasi arac)

Cikti formati: JSON dizisi. Her eleman:
{{
  "input": "[Turkce kullanici talebi]",
  "expected_output": "{{\\"tool\\": \\"...\", \\"params\\": {{...}}}}",
  "difficulty": "easy|medium|hard",
  "metadata": {{"tool": "..."}}
}}""",
    "multi_step": f"""Sen, Turkce cok adimli arac kullanimi degerlendirme veri seti olusturan bir uzmansin.

{MOCK_TOOLS}

Gorevin: Birden fazla araci sirayla kullanmayi gerektiren Turkce senaryolar uretmek.

Kurallar:
- Kullanici bir gorev tanimi verir, bunun icin 2-4 araci sirayla cagirmak gerekir
- Adimlarin mantiksal bir sirasi olmali (bir aracin ciktisi digerinin girdisi olabilir)
- Senaryolar gercekci ve Turkiye baglaminda olmali
- Zorluk: medium (2 adim, acik siralama), hard (3-4 adim, cikarim gerektiren)

Cikti formati: JSON dizisi. Her eleman:
{{
  "input": "[Turkce cok adimli gorev tanimi]",
  "expected_output": "[{{\\"step\\": 1, \\"tool\\": \\"...\\", \\"params\\": {{...}}}}, ...]",
  "difficulty": "medium|hard",
  "metadata": {{"num_steps": N, "tools_used": [...]}}
}}""",
    "parameter_extraction": f"""Sen, Turkce parametre cikarimi degerlendirme veri seti olusturan bir uzmansin.

{MOCK_TOOLS}

Gorevin: Turkce dogal dilden parametre cikarimi ornekleri uretmek.

Kurallar:
- Kullanici belirsiz veya eksik bilgi iceren Turkce bir talep yazar
- Model eksik parametreleri cikarabilmeli veya sormasi gerektigini belirlemeli
- Gunluk konusma dili, argo ve dolayli ifadeler icermeli
- Zorluk: easy (tum parametreler acik), medium (bazi cikarim gerekli), hard (belirsiz/eksik)

Cikti formati: JSON dizisi. Her eleman:
{{
  "input": "[Turkce belirsiz/eksik talep]",
  "expected_output": "{{\\"tool\\": \\"...\\", \\"params\\": {{...}}, \\"missing\\": [...]}}",
  "difficulty": "easy|medium|hard",
  "metadata": {{"tool": "...", "ambiguity_type": "..."}}
}}""",
    "error_recovery": f"""Sen, Turkce hata kurtarma degerlendirme veri seti olusturan bir uzmansin.

{MOCK_TOOLS}

Gorevin: Arac cagrisinda hata olustuğunda modelin nasil tepki vermesi gerektigini olcen ornekler uretmek.

Kurallar:
- Senaryo: kullanici bir talepte bulunur, ilk arac cagirisi hata dondurur
- Model hatayı anlayip alternatif bir yaklasim onerebilmeli
- Hata turleri: gecersiz parametre, servis hatasi, bulunamadi, yetki hatasi
- Zorluk: medium (basit hata, acik alternatif), hard (karmasik hata, yaratici cozum)

Cikti formati: JSON dizisi. Her eleman:
{{
  "input": "Kullanici talebi: [talep]\nHata: [hata mesaji]",
  "expected_output": "[beklenen kurtarma stratejisi ve alternatif cagri]",
  "difficulty": "medium|hard",
  "metadata": {{"error_type": "...", "original_tool": "..."}}
}}""",
}


class ToolGenerator(BaseGenerator):
    category = "tool"

    def _get_subcategories(self) -> list[str]:
        return SUBCATEGORIES

    def _get_system_prompt(self, subcategory: str) -> str:
        return SYSTEM_PROMPTS[subcategory]

    def _get_user_prompt(self, subcategory: str, batch_size: int) -> str:
        return (
            f"{batch_size} adet yeni ve benzersiz Turkce {subcategory} ornegi uret. "
            f"Zorluk dagilimi cesitli olsun. "
            f"Sadece JSON dizisi olarak cevap ver, baska bir sey yazma."
        )
