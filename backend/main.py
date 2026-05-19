import urllib.parse
import google.generativeai as genai
import json
import re
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv

# --- 1. GÜVENLİK VE API KURULUMU ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("KRİTİK HATA: GEMINI_API_KEY .env dosyasında bulunamadı!")

genai.configure(api_key=GEMINI_API_KEY)
generation_config = {"response_mime_type": "application/json"}
scraping_agent = genai.GenerativeModel('gemini-3-flash-preview', generation_config=generation_config)

# --- 2. FASTAPI UYGULAMASI ---
app = FastAPI(title="VTO Sanal Deneme API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. SİSTEM YAPILANDIRMASI (FEATURE FLAGS) ---
DEMO_MODE = False  
DEMO_CATALOG = {
    "amazon.com": {"kasa_capi": "42mm", "materyal": "Çelik", "renk": "Gümüş", "kordon": "Çelik", "resim_url": "https://cdn2.chrono24.com/images/uhren/26053805-7p0w8148bofk4w741p6h5l47-ExtraLarge.jpg", "kaynak": "Demo"},
    "trendyol.com": {"kasa_capi": "40mm", "materyal": "Alüminyum", "renk": "Siyah", "kordon": "Silikon", "resim_url": "https://cdn2.chrono24.com/images/uhren/26053805-7p0w8148bofk4w741p6h5l47-ExtraLarge.jpg", "kaynak": "Demo"},
    "ty.gl": {"kasa_capi": "40mm", "materyal": "Alüminyum", "renk": "Siyah", "kordon": "Silikon", "resim_url": "https://cdn2.chrono24.com/images/uhren/26053805-7p0w8148bofk4w741p6h5l47-ExtraLarge.jpg", "kaynak": "Demo"} 
}

# --- 4. VERİ MODELLERİ ---
class AnalysisRequest(BaseModel):
    gender: str
    height: float
    weight: float
    wristRangeStr: str
    skinColorHex: Optional[str] = "#000000" 
    productLink: str
    manualWatchSize: Optional[str] = None

# EKSİK OLAN MODEL EKLENDİ!
class AlternativeRequest(BaseModel):
    target_style: str
    gender: str
    wristRangeStr: str
    skinColorHex: str

# --- 5. AKILLI ARAÇLAR ---
def extract_from_url(url: str):
    print(f"[Ajan B1] URL analizi başlatıldı: {url}")
    kasa_match = re.search(r'(\d{2})\s*mm', url.lower())
    if kasa_match:
        kasa_capi = f"{kasa_match.group(1)}mm"
        print(f"[Ajan B1] Başarılı! URL içinden kasa çapı kurtarıldı: {kasa_capi}")
        return {"kasa_capi": kasa_capi, "materyal": "Belirtilmemiş", "renk": "Belirtilmemiş", "kordon": "Belirtilmemiş", "kaynak": "URL Analizi (Plan B1)"}
    return None

def agentic_universal_scraper(url: str):
    print(f"\n[Ajan A] İşlem başlatıldı: {url}")
    clean_url = url.split('?')[0].split('/ref=')[0]

    if DEMO_MODE:
        print("[Ajan A] DEMO_MODE aktif. Yerel katalog kontrol ediliyor...")
        for key, data in DEMO_CATALOG.items():
            if key in clean_url:
                return data

    try:
        print(f"[Ajan A] curl_cffi ile Chrome 120 taklidi yapılarak siteye sızılıyor: {clean_url}")
        response = curl_requests.get(clean_url, impersonate="chrome120", timeout=15)

        if response.status_code != 200:
            print(f"[Ajan A] Site erişimi reddetti (HTTP {response.status_code}). B1 Planına geçiliyor...")
            return extract_from_url(clean_url)

        soup = BeautifulSoup(response.content, "html.parser")
        json_ld_data = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                json_ld_data.append(data)
            except:
                continue

        for element in soup(["script", "style", "nav", "footer", "header", "meta", "link", "svg", "path"]):
            element.extract()
        
        clean_text = soup.get_text(separator=" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        optimized_text = clean_text[:4000] 

        context = f"JSON-LD Meta Verisi:\n{json.dumps(json_ld_data)[:1500]}\n\nSayfa Metni:\n{optimized_text}"

        prompt = f"""
        Aşağıdaki metin, bir e-ticaret sitesinden çekilmiş JSON-LD (Schema) verilerini ve sayfanın temizlenmiş metnini içermektedir.
        Senden istediğim bu bağlamı analiz edip, saatin fiziksel özelliklerini bulman ve aşağıdaki anahtarlara sahip JSON objesi döndürmen:
        "kasa_capi" (örneğin: 42mm), "materyal" (örneğin: Çelik), "renk" (örneğin: Siyah), "kordon" (örneğin: Deri), "resim_url" (JSON içindeki image veya fotoğraf linki).
        Eğer bir veriyi kesinlikle bulamazsan değerine "Belirtilmemiş" yaz.
        
        Veri:
        {context}
        """
        print("[Ajan A] Yapay zeka, JSON-LD ve Budanmış HTML'i analiz ediyor...")
        res = scraping_agent.generate_content(prompt)
        
        clean_json_str = res.text.replace('```json', '').replace('```', '').strip()
        extracted_data = json.loads(clean_json_str)
        extracted_data["kaynak"] = "curl_cffi + JSON-LD + Gemini AI"
        
        return extracted_data
    except Exception as e:
        print(f"[Ajan A] Bağlantı/Ayrıştırma Hatası ({e}). B1 Planına geçiliyor...")
        return extract_from_url(clean_url)

def trendyol_search_agent(query: str):
    print(f"\n[Arama Ajanı] Gelen Ham İstek: {query}")
    
    # --- YENİ ZIRH: Tireleri silmek yerine boşluğa çevir (38-40 -> 38 40 olsun) ---
    query = query.replace('-', ' ')
    clean_query = re.sub(r'[^\w\s]', '', query)
    words = clean_query.split()
    
    # --- ZIRH 1: İnsan Gibi Arama Stratejisi (Dar'dan Genişe) ---
    search_terms = []
    if len(words) >= 3:
        search_terms.append(f"{words[0]} {words[1]} {words[2]} saat") # Örn: 39mm çelik spor saat
        search_terms.append(f"{words[0]} {words[1]} saat")            # Örn: 39mm çelik saat
        search_terms.append(f"{words[1]} saat")                       # Örn: çelik saat (Kesin bulur)
    elif len(words) == 2:
        search_terms.append(f"{words[0]} {words[1]} saat")
        search_terms.append(f"{words[1]} saat")
    else:
        search_terms.append(f"{clean_query} saat")
        
    for term in search_terms:
        print(f"[Arama Ajanı] Trendyol'da deneniyor: {term}")
        safe_query = urllib.parse.quote_plus(term)
        search_url = f"https://www.trendyol.com/sr?q={safe_query}"
        
        try:
            response = curl_requests.get(search_url, impersonate="chrome120", timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # --- ZIRH 2: Class İsimlerini Boşver, Doğrudan Linki (Regex ile) Avla! ---
                # Trendyol ürün linkleri HER ZAMAN "-p-" ve ardından ürün numarası içerir (Örn: -p-123456)
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if re.search(r'-p-\d+', href) and "/yorumlar" not in href:
                        # Eğer link "http" ile başlamıyorsa, ana domaini ekle
                        if not href.startswith("http"):
                            href = "https://www.trendyol.com" + href
                        
                        print(f"[Arama Ajanı] Ürün BAŞARIYLA AVLANDI: {href}")
                        return href # İlk bulduğu gerçek ürün linkini döndür ve savaşı bitir!
                        
        except Exception as e:
            print(f"[Arama Ajanı] Ağ hatası: {e}")
            
        print(f"[Arama Ajanı] '{term}' kelimesiyle sonuç çıkmadı, ağ genişletiliyor...")
        
    print("[Arama Ajanı] Hiçbir arama stratejisi işe yaramadı.")
    return None

# --- 6. API ENDPOINT'İ (PAYLAŞIMLI BAĞLAM MİMARİSİ) ---
@app.post("/analyze")
async def analyze_watch(data: AnalysisRequest):
    print(f"\n--- YENİ SANAL DENEME İSTEĞİ ({data.wristRangeStr} Bilek) ---")
    
    # 1. ADIM: PAYLAŞIMLI BAĞLAM (SHARED CONTEXT) OLUŞTURULUYOR
    # Tüm ajanların bakacağı "Tek Gerçeklik Kaynağı" (Ground Truth)
    shared_context = {
        "user_anatomy": {
            "cinsiyet": "Erkek" if data.gender == "male" else "Kadın",
            "bilek": data.wristRangeStr,
            "ten_rengi": data.skinColorHex if data.skinColorHex else 'Belirtilmemiş',
            "manuel_mudahale": data.manualWatchSize
        },
        "watch_data": None,
        "stylist_report": None,
        "system_status": "initializing"
    }
    print("[Sistem] Paylaşımlı Bağlam (Shared Context) başarıyla oluşturuldu.")

    # 2. ADIM: KAZIMA AJANI (SCRAPER AGENT) BAĞLAMI GÜNCELLER
    print("[Kazıma Ajanı] Bağlamdaki veriler okunuyor ve web'den ürün verisi çekiliyor...")
    if shared_context["user_anatomy"]["manuel_mudahale"]:
        shared_context["watch_data"] = {
            "kasa_capi": shared_context["user_anatomy"]["manuel_mudahale"],
            "materyal": "Bilinmiyor",
            "renk": "Bilinmiyor",
            "kordon": "Bilinmiyor",
            "kaynak": "Kullanıcı Manuel Girişi (Plan B2)"
        }
    else:
        shared_context["watch_data"] = agentic_universal_scraper(data.productLink)

    # Güvenlik Duvarı veya Veri Çökmesi Kontrolü
    kasa_capi = shared_context["watch_data"].get("kasa_capi") if shared_context["watch_data"] else None
    if not kasa_capi or kasa_capi == "Belirtilmemiş" or kasa_capi is None:
        print("[Sistem] Otonom veri çekilemedi. Plan B2 (Manuel) devreye giriyor.")
        return {
            "status": "manual_input_needed",
            "message": "Güvenlik duvarları nedeniyle saatin ölçülerini otomatik okuyamadık. Lütfen kasa çapını aşağıdan seçin."
        }

    # 3. ADIM: STİLİST AJAN BAĞLAMI OKUYOR VE YORUM YAPIYOR
    print("[Stilist Ajan] Bağlam havuzundan kullanıcı ve ürün verileri sentezleniyor...")
    
    prompt_stylist = f"""
    Sen lüks saatler ve stil konusunda fütüristik bir Yapay Zeka Asistanısın.

    KULLANICI: Cinsiyet: {shared_context['user_anatomy']['cinsiyet']}, Bilek: {shared_context['user_anatomy']['bilek']}, Ten (Hex): {shared_context['user_anatomy']['ten_rengi']}
    SAAT: Çap: {shared_context['watch_data'].get('kasa_capi', 'Bilinmiyor')}, Materyal: {shared_context['watch_data'].get('materyal', 'Bilinmiyor')}

    KURALLAR:
    1. YORUM YAPISI: Kullanıcıya rozet destekli mikro satırlar halinde bir analiz yaz. HTML kullanarak her satırın başına aşağıdaki gibi kapsüller ekle ve aralarına <br><br> koy:
       <span style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; letter-spacing: 1px; margin-right: 10px; vertical-align: middle;'>FORM</span> [Saatin tarzı ve formu üzerine tek cümle]
       <br><br>
       <span style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; letter-spacing: 1px; margin-right: 10px; vertical-align: middle;'>UYUM</span> [Bilek ölçüsü ve kasa çapı uyumu üzerine tek cümle]
       <br><br>
       <span style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; letter-spacing: 1px; margin-right: 10px; vertical-align: middle;'>TON</span> [Saat materyali ile doğal ten renginin alt ton uyumu üzerine tek cümle]
       ⚠️ KRİTİK KURAL: Metin içinde KESİNLİKLE ham renk kodlarını (#DBA399 vb.) yazma! Yerine 'buğday ten', 'açık ten' gibi kelimeler kullan. JSON hatası almamak için metinde çift tırnak (") ASLA kullanma, sadece tek tırnak (') kullan. 
    2. CİNSİYET UYARISI: Eğer kullanıcı "Kadın" ise ve seçilen saat bariz bir erkek saatiyse SADECE "Bu saat erkek saatidir." yaz. Eğer cinsiyet uyumsuzluğu yoksa "warning" değerini null yap. Cinsiyet uyumsuzluğu varsa skoru KESİNLİKLE 45'in altında tut.
    3. VURGU: Önemli kelimeleri <span style='color: #FFFFFF; font-weight: bold; text-shadow: 0 0 5px rgba(255,255,255,0.3);'> kelime </span> ile vurgula.
    4. ÖNERİLER: Sistemin Trendyol'da otomatik arama yapabilmesi için MAKSİMUM 3 KELİMELİK, ÇOK KISA e-ticaret arama terimleri üret (Örn: '38mm çelik klasik', '40mm siyah deri'). KESİNLİKLE uzun cümleler, tire (-) veya "aralığında", "esintili" gibi kelimeler KULLANMA!    
    ÇIKTI FORMATI (SADECE JSON):
    {{
        "match_score": <10 ile 100 arası>,
        "warning": "<Sadece kısa uyarı metni veya null>",
        "stylist_comment": "<Detaylı HTML metin>",
        "recommendations": ["<Öneri 1>", "<Öneri 2>"]
    }}
    """

    try:
        res_stylist = scraping_agent.generate_content(prompt_stylist)
        clean_stylist_json = res_stylist.text.replace('```json', '').replace('```', '').strip()
        
        # --- TIP KORUMA ZIRHI: Gelen veri liste ise sözlüğe güvenli bir şekilde ayıkla ---
        parsed_json = json.loads(clean_stylist_json)
        if isinstance(parsed_json, list) and len(parsed_json) > 0:
            parsed_json = parsed_json[0]  # Listenin içindeki ilk gerçek sözlük nesnesini al
        if not isinstance(parsed_json, dict):
            parsed_json = {}
            
        shared_context["stylist_report"] = parsed_json
        
        # 2. İNCELEME AJANI (GUARDRAIL) BAĞLAM ÜZERİNDEN DENETLİYOR
        print("[İnceleme Ajanı] Çıktı doğrulanıyor...")
        if '"' in shared_context["stylist_report"].get("stylist_comment", ""):
            print("[İnceleme Ajanı] HATA: Çıktıda yasaklı çift tırnak bulundu! Düzeltiliyor...")
            shared_context["stylist_report"]["stylist_comment"] = shared_context["stylist_report"]["stylist_comment"].replace('"', "'")
        
        if shared_context["stylist_report"].get("match_score") is None:
            shared_context["stylist_report"]["match_score"] = 70
            
        print(f"[İnceleme Ajanı] Doğrulama başarılı. Skor: %{shared_context['stylist_report'].get('match_score')}")
        
    except Exception as e:
        print(f"[Stilist Ajan] Hata oluştu: {e}")
        # LLM patlarsa veya beklenmedik bir yapı dönerse devreye giren "Fallback" mekanizması
        shared_context["stylist_report"] = {
            "match_score": 75,
            "warning": None,
            "stylist_comment": "Saat modeliniz tarzınıza şık bir dokunuş katacaktır ancak tam uyumluluk analizi için ek görsel verilere ihtiyaç var.",
            "recommendations": ["38mm çelik klasik", "40mm siyah deri"]
        }

    shared_context["system_status"] = "success"

    return {
        "status": shared_context["system_status"],
        "message": "Bağlam mimarisi başarıyla tamamlandı.",
        "scraped_data": shared_context["watch_data"],
        "stylist_data": shared_context["stylist_report"] 
    }

# --- 7. ALTERNATİF GERÇEK ZAMANLI ARAMA ENDPOINT'İ (PAYLAŞIMLI BAĞLAM) ---
@app.post("/simulate_alternative")
async def simulate_alt(data: AlternativeRequest):
    print(f"\n--- ALTERNATİF ROTA İSTEĞİ: {data.target_style} ---")
    
    # 1. PAYLAŞIMLI BAĞLAM (SHARED CONTEXT) OLUŞTURULUYOR
    shared_context = {
        "user_anatomy": {
            "cinsiyet": "Erkek" if data.gender == "male" else "Kadın",
            "bilek": data.wristRangeStr,
            "ten_rengi": data.skinColorHex
        },
        "target_style": data.target_style,
        "watch_data": None,
        "stylist_report": None
    }
    print("[Sistem] Alternatif rota için Paylaşımlı Bağlam oluşturuldu.")
    
    arama_terimi = f"{shared_context['target_style']} {shared_context['user_anatomy']['cinsiyet']}"
    gercek_urun_linki = trendyol_search_agent(arama_terimi)
    
    if not gercek_urun_linki:
        return {"status": "error", "message": "Bu tarza uygun ürün bulunamadı."}
        
    print("[Veri Ajanı] Gerçek ürün verisi bağlama aktarılıyor...")
    shared_context["watch_data"] = agentic_universal_scraper(gercek_urun_linki)
    
    # --- YENİ EKLENEN KORUMA KALKANI (KOTA DOLMASI DURUMUNDA) ---
    if not shared_context["watch_data"]: 
        shared_context["watch_data"] = {
            "kasa_capi": "Bilinmiyor",
            "materyal": "Bilinmiyor",
            "resim_url": "Belirtilmemiş"
        }
    # -----------------------------------
    
    print("[Stilist Ajan] Alternatif model analiz ediliyor...")
    prompt_alt = f"""
    Sen lüks saatler ve stil konusunda fütüristik bir Yapay Zeka Asistanısın.
    
    KULLANICI: Cinsiyet: {shared_context['user_anatomy']['cinsiyet']}, Bilek: {shared_context['user_anatomy']['bilek']}, Ten: {shared_context['user_anatomy']['ten_rengi']}
    SİSTEMİN BULDUĞU YENİ SAAT: Çap: {shared_context['watch_data'].get('kasa_capi', 'Bilinmiyor')}, Materyal: {shared_context['watch_data'].get('materyal', 'Bilinmiyor')}
    
    Kullanıcı "{shared_context['target_style']}" tarzına tıkladı ve sistem ona gerçek zamanlı olarak yukarıdaki saati buldu.
    
    KURALLAR:
    1. YORUM: Kullanıcıya bu yeni saatin neden onun için KUSURSUZ (veya çok daha iyi) bir seçim olduğunu 3-4 cümleyle anlat. 
    2. VURGU: Önemli kelimeleri <span style='color: #FFFFFF; font-weight: bold; text-shadow: 0 0 5px rgba(255,255,255,0.3);'> kelime </span> ile vurgula.
    3. SKOR: Bu saat özel seçildiği için match_score KESİNLİKLE 85 ile 100 arasında olmalı.
    4. Çıktın KESİNLİKLE geçerli bir JSON olmalıdır ve içinde çift tırnak (") KULLANILMAMALIDIR (tek tırnak kullan).
    
    ÇIKTI FORMATI (SADECE JSON):
    {{
        "match_score": <85 ile 100 arası>,
        "warning": null,
        "stylist_comment": "<Detaylı HTML metin>",
        "recommendations": []
    }}
    """
    
    try:
        res = scraping_agent.generate_content(prompt_alt)
        clean_json = res.text.replace('```json', '').replace('```', '').strip()
        
        # --- TIP KORUMA ZIRHI ---
        parsed_alt = json.loads(clean_json)
        if isinstance(parsed_alt, list) and len(parsed_alt) > 0:
            parsed_alt = parsed_alt[0]
        if not isinstance(parsed_alt, dict):
            parsed_alt = {}
            
        shared_context["stylist_report"] = parsed_alt
        
        print("[İnceleme Ajanı] Çıktı doğrulanıyor...")
        if '"' in shared_context["stylist_report"].get("stylist_comment", ""):
            print("[İnceleme Ajanı] HATA: Çıktıda yasaklı çift tırnak bulundu! Düzeltiliyor...")
            shared_context["stylist_report"]["stylist_comment"] = shared_context["stylist_report"]["stylist_comment"].replace('"', "'")
        
        if shared_context["stylist_report"].get("match_score") is None:
            shared_context["stylist_report"]["match_score"] = 70
            
        print(f"[İnceleme Ajanı] Doğrulama başarılı. Skor: %{shared_context['stylist_report'].get('match_score')}")
        
    except Exception as e:
        print(f"[Ajan] Alternatif üretilirken hata: {e}")
        return {"status": "error", "message": str(e)}
        
    # GÜVENLİ RETURN: except bloğunun DIŞINDA, yani try-except ile AYNI HİZADA olmalı!
    return {
        "status": "success",
        "scraped_data": shared_context.get("watch_data"), 
        "stylist_data": shared_context.get("stylist_report")    
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)