from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import re

# --- UYGULAMA VE YAPAY ZEKA KURULUMU ---
app = FastAPI(title="VTO Sanal Deneme API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Hackathon öncesi gerçek Gemini API anahtarını buraya eklemelisin
GEMINI_API_KEY = "BURAYA_API_ANAHTARINI_YAZ"
genai.configure(api_key=GEMINI_API_KEY)

# Gemini'ın sadece JSON formatında, makine okuyabilir yanıt vermesini zorluyoruz
generation_config = {"response_mime_type": "application/json"}
scraping_agent = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)

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
    """
    PLAN A (Otonom Madenci): Siteye gider, ham HTML metnini alır ve Gemini'a JSON olarak ayıklatır.
    """
    print(f"\n[Ajan A] Hedef siteye gidiliyor: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    try:
        response = requests.get(url, headers=headers, timeout=8)
        
        # Anti-Bot Korumasına Takılırsak Plan B1'e geç
        if response.status_code != 200:
            print(f"[Ajan A] Site botu engelledi (HTTP {response.status_code}). B1 Planına geçiliyor...")
            return extract_from_url(url)

        # BS4 ile DOM'u temizle ve sadece metni al
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "footer", "header", "nav"]):
            script.extract()
            
        raw_text = soup.get_text(separator=" ", strip=True)
        optimized_text = raw_text[:4000] # API Token tasarrufu için metni kırp

        if GEMINI_API_KEY == "BURAYA_API_ANAHTARINI_YAZ":
            print("[Ajan Uyarısı] Gerçek API Anahtarı eksik, otonom kazıma yapılamıyor. B1 Planına geçiliyor...")
            return extract_from_url(url)

        # Gemini Veri Çıkarma Komutu
        prompt = f"""
        Aşağıdaki karmaşık web sitesi metninin içinde bir saat satılmaktadır. 
        Senden istediğim bu metni analiz edip, saatin fiziksel özelliklerini bulman ve aşağıdaki anahtarlara sahip JSON objesi döndürmen:
        "kasa_capi" (örneğin: 42mm), "materyal" (örneğin: Çelik), "renk" (örneğin: Siyah), "kordon" (örneğin: Deri).
        Eğer bir veriyi metinde bulamazsan değerine "Belirtilmemiş" yaz. Sadece Kasa Çapını bulman bile yeterlidir.
        
        Web Sitesi Metni:
        {optimized_text}
        """

        print("[Ajan A] Yapay zeka metin yığınını analiz ediyor...")
        res = scraping_agent.generate_content(prompt)
        
        # Gemini'dan dönen metin markdown backtick'leri (```json ... ```) içeriyorsa temizle
        clean_json_str = res.text.replace('```json', '').replace('```', '').strip()
        extracted_data = json.loads(clean_json_str)
        extracted_data["kaynak"] = "Gemini AI (Plan A)"
        
        print(f"[Ajan A] Veri başarıyla ayıklandı: {extracted_data}")
        return extracted_data

    except Exception as e:
        print(f"[Ajan A] Beklenmeyen Hata ({e}). B1 Planına geçiliyor...")
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