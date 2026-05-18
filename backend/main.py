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
    
    # Özel karakterleri temizle
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

# --- 6. API ENDPOINT'İ ---
@app.post("/analyze")
async def analyze_watch(data: AnalysisRequest):
    print(f"\n--- YENİ SANAL DENEME İSTEĞİ ({data.wristRangeStr} Bilek) ---")
    
    if data.manualWatchSize:
        watch_features = {
            "kasa_capi": data.manualWatchSize, "materyal": "Bilinmiyor", "renk": "Bilinmiyor",
            "kordon": "Bilinmiyor", "kaynak": "Kullanıcı Manuel Girişi (Plan B2)"
        }
    else:
        watch_features = agentic_universal_scraper(data.productLink)

    kasa_capi = watch_features.get("kasa_capi") if watch_features else None
    
    if not kasa_capi or kasa_capi == "Belirtilmemiş" or kasa_capi is None:
        return {
            "status": "manual_input_needed",
            "message": "Güvenlik duvarları nedeniyle saatin ölçülerini otomatik okuyamadık. Analiz için lütfen kasa çapını aşağıdan seçin."
        }

    print("\n[Stilist Ajan] Yapay Zeka kişiselleştirilmiş moda yorumu üretiyor...")
    cinsiyet_tr = "Erkek" if data.gender == "male" else "Kadın"

    prompt_stylist = f"""
    Sen lüks saatler ve stil konusunda fütüristik bir Yapay Zeka Asistanısın.

    KULLANICI: Cinsiyet: {cinsiyet_tr}, Bilek: {data.wristRangeStr}, Ten (Hex): {data.skinColorHex if data.skinColorHex else 'Belirtilmemiş'}
    SAAT: Çap: {watch_features.get('kasa_capi', 'Bilinmiyor')}, Materyal: {watch_features.get('materyal', 'Bilinmiyor')}

    KURALLAR:
    1. YORUM YAPISI: Kullanıcıya rozet destekli mikro satırlar halinde bir analiz yaz. HTML kullanarak her satırın başına aşağıdaki gibi kapsüller ekle ve aralarına <br><br> koy:
       <span style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; letter-spacing: 1px; margin-right: 10px; vertical-align: middle;'>FORM</span> [Saatin tarzı ve formu üzerine tek cümle]
       <br><br>
       <span style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; letter-spacing: 1px; margin-right: 10px; vertical-align: middle;'>UYUM</span> [Bilek ölçüsü ve kasa çapı uyumu üzerine tek cümle]
       <br><br>
       <span style='background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; letter-spacing: 1px; margin-right: 10px; vertical-align: middle;'>TON</span> [Saat materyali ile doğal ten renginin alt ton uyumu üzerine tek cümle]
    2. CİNSİYET UYARISI: Uyumsuzluk varsa 'warning' değerine sadece 'Bu saat kadın/erkek saatidir.' yaz ve match_score değerini KESİNLİKLE 45'in altında tut. Yoksa null yap.
    3. VURGU: Önemli kelimeleri <span style='color: #FFFFFF; font-weight: bold; text-shadow: 0 0 5px rgba(255,255,255,0.3);'> kelime </span> ile vurgula.
    4. ÖNERİLER: Sistemin e-ticaret sitesinde otomatik arama yapabilmesi için MAKSİMUM 3-4 KELİMELİK çok net ve kısa ARAMA KELİMELERİ belirle (Örn: '38mm çelik klasik', '40mm siyah deri', 'titanyum spor'). Kesinlikle uzun cümleler veya şiirsel betimlemeler kullanma!    
    5. JSON KORUMASI: Metinlerde ASLA çift tırnak (") kullanma! Tek tırnak (') kullan.
    
    ÇIKTI FORMATI (SADECE JSON):
    {{
        "match_score": <10 ile 100 arası>,
        "warning": "<Sadece kısa uyarı metni veya null>",
        "stylist_comment": "<Detaylı ve vurgulu HTML metin>",
        "recommendations": ["<Öneri 1>", "<Öneri 2>"]
    }}
    """

    try:
        res_stylist = scraping_agent.generate_content(prompt_stylist)
        clean_stylist_json = res_stylist.text.replace('```json', '').replace('```', '').strip()
        stylist_data = json.loads(clean_stylist_json)
        print(f"[Stilist Ajan] Başarılı! Skor: %{stylist_data.get('match_score')}")
    except Exception as e:
        print(f"[Stilist Ajan] Hata oluştu: {e}")
        stylist_data = {"match_score": 75, "stylist_comment": "Analiz için ek görsel verilere ihtiyaç var."}

    return {"status": "success", "message": "Başarılı.", "scraped_data": watch_features, "stylist_data": stylist_data}

# --- 7. ALTERNATİF GERÇEK ZAMANLI ARAMA ENDPOINT'İ ---
@app.post("/simulate_alternative")
async def simulate_alt(data: AlternativeRequest):
    print(f"\n--- ALTERNATİF ROTA İSTEĞİ: {data.target_style} ---")
    
    cinsiyet_tr = "Erkek" if data.gender == "male" else "Kadın"
    arama_terimi = f"{data.target_style} {cinsiyet_tr}"
    
    gercek_urun_linki = trendyol_search_agent(arama_terimi)
    
    if not gercek_urun_linki:
        return {"status": "error", "message": "Bu tarza uygun ürün bulunamadı."}
        
    watch_features = agentic_universal_scraper(gercek_urun_linki)
    
    prompt_alt = f"""
    Sen lüks saatler ve stil konusunda fütüristik bir Yapay Zeka Asistanısın.
    KULLANICI: Cinsiyet: {cinsiyet_tr}, Bilek: {data.wristRangeStr}, Ten (Hex): {data.skinColorHex}
    YENİ SAAT: Çap: {watch_features.get('kasa_capi', 'Bilinmiyor')}, Materyal: {watch_features.get('materyal', 'Bilinmiyor')}
    
    Kullanıcı "{data.target_style}" tarzına tıkladı ve ona bu gerçek saat bulundu.
    
    KURALLAR:
    1. YORUM: Neden onun için KUSURSUZ bir seçim olduğunu 3-4 cümleyle anlat. 
    2. VURGU: Önemli kelimeleri <span style='color: #FFFFFF; font-weight: bold; text-shadow: 0 0 5px rgba(255,255,255,0.3);'> kelime </span> ile vurgula.
    3. SKOR: match_score KESİNLİKLE 85 ile 100 arasında olmalı.
    4. Çift tırnak (") kullanma.
    
    ÇIKTI FORMATI (SADECE JSON):
    {{
        "match_score": <85 ile 100 arası>,
        "warning": null,
        "stylist_comment": "<Detaylı ve vurgulu HTML metin>",
        "recommendations": []
    }}
    """
    
    try:
        res = scraping_agent.generate_content(prompt_alt)
        clean_json = res.text.replace('```json', '').replace('```', '').strip()
        stylist_data = json.loads(clean_json)
        return {"status": "success", "scraped_data": watch_features, "stylist_data": stylist_data}
    except Exception as e:
        print(f"[Ajan] Alternatif üretilirken hata: {e}")
        return {"status": "error"}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)