"""System prompts for evaluation — instruct models on expected response format."""

TOOL_DEFINITIONS = """Kullanılabilir araçlar (tools):

1. hava_durumu(sehir: str, tarih: str | None) -> {"sicaklik": float, "durum": str, "nem": float}
2. restoran_ara(konum: str, mutfak: str | None, min_puan: float | None) -> [{"ad": str, "puan": float, "adres": str}]
3. ucus_ara(kalkis: str, varis: str, tarih: str, yolcu_sayisi: int) -> [{"sefer": str, "saat": str, "fiyat": float}]
4. doviz_kuru(kaynak: str, hedef: str, miktar: float) -> {"sonuc": float, "kur": float}
5. haber_getir(kategori: str, adet: int) -> [{"baslik": str, "ozet": str, "tarih": str}]
6. takvim_olustur(baslik: str, tarih: str, saat: str, sure_dk: int, katilimcilar: list[str] | None) -> {"id": str, "durum": str}
7. mesaj_gonder(alici: str, icerik: str, oncelik: str | None) -> {"durum": str, "id": str}
8. dosya_ara(sorgu: str, dosya_tipi: str | None, klasor: str | None) -> [{"ad": str, "yol": str, "boyut": int}]"""

SYSTEM_PROMPTS: dict[str, dict[str, str]] = {
    "core": {
        "reading_comprehension": (
            "Verilen metni dikkatlice oku ve soruyu yanıtla. "
            "Sadece cevabı yaz, açıklama yapma."
        ),
        "common_sense": (
            "Çoktan seçmeli soruyu yanıtla. "
            "Sadece doğru şıkkın harfini ve metnini yaz (örnek: B) Cevap metni). "
            "Açıklama ekleme."
        ),
        "grammar": (
            "Verilen dil bilgisi görevini yerine getir. "
            "Sadece cevabı yaz, açıklama yapma."
        ),
        "translation": (
            "Verilen metni çevir. Sadece çeviriyi yaz, başka bir şey ekleme."
        ),
        "summarization": (
            "Verilen metni özetle. Sadece özeti yaz, başka bir şey ekleme."
        ),
    },
    "tool": {
        "api_call": (
            f"{TOOL_DEFINITIONS}\n\n"
            "Kullanıcının talebine göre doğru aracı doğru parametrelerle çağır. "
            'Sadece JSON formatında yanıt ver: {"tool": "araç_adı", "params": {...}}'
        ),
        "multi_step": (
            f"{TOOL_DEFINITIONS}\n\n"
            "Kullanıcının talebini yerine getirmek için gereken araç çağrılarını sırasıyla belirle. "
            'JSON dizisi formatında yanıt ver: [{"tool": "...", "params": {...}}, ...]'
        ),
        "parameter_extraction": (
            f"{TOOL_DEFINITIONS}\n\n"
            "Kullanıcının talebinden araç çağrısı parametrelerini çıkar. "
            'Sadece JSON formatında yanıt ver: {"tool": "araç_adı", "params": {...}}'
        ),
        "error_recovery": (
            f"{TOOL_DEFINITIONS}\n\n"
            "Verilen hatalı araç çağrısını analiz et ve düzeltilmiş versiyonunu yaz. "
            "Hatayı ve çözümü açıkla."
        ),
    },
    "fin": {
        "document_comprehension": (
            "Verilen finansal metni analiz et ve soruyu yanıtla. "
            "Sadece cevabı yaz, kısa ve öz ol."
        ),
        "sentiment": (
            "Verilen finansal metnin duygusunu belirle. "
            "Sadece şu kelimelerden birini yaz: pozitif, negatif veya nötr. "
            "Başka bir şey yazma."
        ),
        "numerical_reasoning": (
            "Verilen finansal verileri analiz et ve soruyu yanıtla. "
            "Sadece cevabı yaz, hesaplama adımlarını gösterme."
        ),
        "term_understanding": (
            "Verilen finansal terimi veya kavramı açıkla. "
            "Kısa ve öz bir tanım yaz."
        ),
    },
    "legal": {
        "document_comprehension": (
            "Verilen hukuki metni analiz et ve soruyu yanıtla. "
            "Sadece cevabı yaz, kısa ve öz ol."
        ),
        "case_reasoning": (
            "Verilen hukuki durumu analiz et ve soruyu yanıtla. "
            "Sadece cevabı yaz, kısa ve öz ol."
        ),
        "clause_extraction": (
            "Verilen metinden istenen maddeyi veya hükmü çıkar. "
            "Sadece ilgili bölümü yaz."
        ),
        "regulatory_compliance": (
            "Verilen durumun mevzuata uygunluğunu değerlendir. "
            "Sadece cevabı yaz, kısa ve öz ol."
        ),
    },
}


def get_system_prompt(category: str, subcategory: str) -> str:
    """Return the system prompt for a given category/subcategory eval."""
    cat_prompts = SYSTEM_PROMPTS.get(category, {})
    return cat_prompts.get(subcategory, "Soruyu kısa ve öz yanıtla.")
