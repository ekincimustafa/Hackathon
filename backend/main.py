import google.generativeai as genai
import json
import re
import os
import uvicorn
import requests
import cloudscraper
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from dotenv import load_dotenv


# Gizli .env dosyasındaki verileri sisteme yükle
load_dotenv()

# API Key'i çevre değişkeninden güvenli bir şekilde çek
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Eğer anahtar bulunamazsa sistemi uyararak durdur (Güvenlik önlemi)
if not GEMINI_API_KEY:
    raise ValueError("KRİTİK HATA: GEMINI_API_KEY .env dosyasında bulunamadı!")

genai.configure(api_key=GEMINI_API_KEY)

# --- UYGULAMA VE YAPAY ZEKA KURULUMU ---
app = FastAPI(title="VTO Sanal Deneme API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini'ın sadece JSON formatında, makine okuyabilir yanıt vermesini zorluyoruz
generation_config = {"response_mime_type": "application/json"}
scraping_agent = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)

# Sunum anında veya testlerde risk almamak için Feature Flag
DEMO_MODE = True  
DEMO_CATALOG = {
    "amazon.com": {"kasa_capi": "42mm", "materyal": "Paslanmaz Çelik", "renk": "Gümüş", "kordon": "Çelik", "kaynak": "Feature Flag (Demo Modu)"},
    "trendyol.com": {"kasa_capi": "40mm", "materyal": "Alüminyum", "renk": "Siyah", "kordon": "Silikon", "kaynak": "Feature Flag (Demo Modu)"}
}

# --- VERİ MODELLERİ ---
class AnalysisRequest(BaseModel):
    gender: str
    height: float
    weight: float
    wristRangeStr: str
    skinColorHex: str
    productLink: str
    # Fallback UI (B2 Planı) devredeyken frontend'den gelecek opsiyonel veri:
    manualWatchSize: str = None 

# --- AKILLI ARAÇLAR (HATA YÖNETİMİ VE SCRAPING) ---

def extract_from_url(url: str):
    """
    PLAN B1 (Zarif Çöküş): Site erişimi engellerse URL metninin içini Regex ile analiz eder.
    Örn: ".../casio-erkek-saat-42mm-p-123" linkinden '42mm' verisini bulup çıkarır.
    """
    print(f"[Ajan B1] URL analizi başlatıldı: {url}")
    url_lower = url.lower()
    
    # URL içinde 2 basamaklı sayı ve yanında 'mm' arar (Örn: 42mm, 38 mm)
    kasa_match = re.search(r'(\d{2})\s*mm', url_lower)
    
    if kasa_match:
        kasa_capi = f"{kasa_match.group(1)}mm"
        print(f"[Ajan B1] Başarılı! URL içinden kasa çapı kurtarıldı: {kasa_capi}")
        return {
            "kasa_capi": kasa_capi,
            "materyal": "Belirtilmemiş",
            "renk": "Belirtilmemiş",
            "kordon": "Belirtilmemiş",
            "kaynak": "URL Analizi (Plan B1)"
        }
    
    print("[Ajan B1] URL içinde kasa çapı bulunamadı.")
    return None

def agentic_universal_scraper(url: str):
    print(f"\n[Ajan A] İşlem başlatıldı: {url}")
    clean_url = url.split('?')[0].split('/ref=')[0]

    # 1. Feature Flag (Demo Modu) Kontrolü
    if DEMO_MODE:
        print("[Ajan A] DEMO_MODE aktif. Yerel katalog kontrol ediliyor...")
        for key, data in DEMO_CATALOG.items():
            if key in clean_url:
                print("[Ajan A] Demo verisi başarıyla yüklendi.")
                return data

    # 2. Jina Reader API (Yeni Nesil LLM Kazıyıcı)
    print("[Ajan A] Jina Reader API ile siteye erişiliyor...")
    jina_url = f"https://r.jina.ai/{clean_url}"

    try:
        # Jina API'ye istek atıyoruz
        response = requests.get(jina_url, timeout=15)

        if response.status_code == 200 and len(response.text) > 100:
            optimized_text = response.text[:4000]

            if GEMINI_API_KEY == "BURAYA_API_ANAHTARINI_YAZ":
                print("[Ajan Uyarısı] Gerçek Gemini API Anahtarı eksik. B1 Planına geçiliyor...")
                return extract_from_url(url)

            prompt = f"""
            Aşağıdaki metin, Jina Reader ile çekilmiş bir e-ticaret sayfasıdır. 
            Senden istediğim bu metni analiz edip, saatin fiziksel özelliklerini bulman ve aşağıdaki anahtarlara sahip JSON objesi döndürmen:
            "kasa_capi" (örneğin: 42mm), "materyal" (örneğin: Çelik), "renk" (örneğin: Siyah), "kordon" (örneğin: Deri).
            Eğer bir veriyi metinde bulamazsan değerine "Belirtilmemiş" yaz.
            
            Sayfa Metni:
            {optimized_text}
            """

            res = scraping_agent.generate_content(prompt)
            clean_json_str = res.text.replace('```json', '').replace('```', '').strip()
            extracted_data = json.loads(clean_json_str)
            extracted_data["kaynak"] = "Jina Reader + Gemini AI"
            
            print(f"[Ajan A] Veri başarıyla ayıklandı: {extracted_data}")
            return extracted_data
            
        else:
            print(f"[Ajan A] Jina Reader başarısız oldu (HTTP {response.status_code}). B1 Planına geçiliyor...")
            return extract_from_url(url)

    except Exception as e:
        print(f"[Ajan A] Bağlantı Hatası ({e}). B1 Planına geçiliyor...")
        return extract_from_url(url)

# --- ANA API ENDPOINT'İ ---

@app.post("/analyze")
async def analyze_watch(data: AnalysisRequest):
    print(f"\n--- YENİ SANAL DENEME İSTEĞİ ({data.wristRangeStr} Bilek) ---")
    
    # Kullanıcı Frontend'deki Fallback UI üzerinden manuel seçim yapmışsa kazımayı atla
    if data.manualWatchSize:
        print(f"Kullanıcı manuel müdahalesi algılandı: {data.manualWatchSize}")
        watch_features = {
            "kasa_capi": data.manualWatchSize,
            "materyal": "Bilinmiyor",
            "renk": "Bilinmiyor",
            "kordon": "Bilinmiyor",
            "kaynak": "Kullanıcı Manuel Girişi (Plan B2)"
        }
    else:
        # Önce Plan A, başarısız olursa Plan B1 otonom olarak çalışır
        watch_features = agentic_universal_scraper(data.productLink)

    # EĞER Plan A ve Plan B1 kasa çapını bulamamışsa (Tam Başarısızlık Durumu)
    # Dürüst UX (Plan B2) sinyali gönder: Frontend'e "Bana manuel bir buton ekranı aç" diyoruz.
    kasa_capi = watch_features.get("kasa_capi") if watch_features else None
    
    if not kasa_capi or kasa_capi == "Belirtilmemiş" or kasa_capi is None:
        print("Tüm otonom veri çekme planları başarısız. Kullanıcıya danışılıyor (Plan B2)...")
        return {
            "status": "manual_input_needed",
            "message": "Güvenlik duvarları nedeniyle saatin ölçülerini otomatik okuyamadık. Analiz için lütfen kasa çapını aşağıdan seçin."
        }

    # TODO: Bu adımda 'watch_features' ve 'data' kullanılarak Stilizasyon (Uygunluk) Ajanı çalıştırılacak.
    
    # Şimdilik başarı durumunda frontend'in verileri alıp almadığını kontrol için geri dönüyoruz
    return {
        "status": "success",
        "message": "Model özellikleri başarıyla işlendi.",
        "scraped_data": watch_features
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)