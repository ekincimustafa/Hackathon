import google.generativeai as genai
import json
import re
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests # YENİ SİLAHIMIZ: TLS Parmak İzi Taklitçisi
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
    skinColorHex: str
    productLink: str
    manualWatchSize: str = None 

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
    """
    PLAN A (curl_cffi + JSON-LD): Chrome tarayıcısının TLS parmak izini kopyalayarak
    WAF (Cloudflare/Datadome) engellerini aşar. Sitenin Schema.org verisini ve 
    budanmış HTML'ini LLM'e sunar.
    """
    print(f"\n[Ajan A] İşlem başlatıldı: {url}")
    clean_url = url.split('?')[0].split('/ref=')[0]

    if DEMO_MODE:
        print("[Ajan A] DEMO_MODE aktif. Yerel katalog kontrol ediliyor...")
        for key, data in DEMO_CATALOG.items():
            if key in clean_url:
                print("[Ajan A] Demo verisi başarıyla yüklendi.")
                return data

    try:
        print(f"[Ajan A] curl_cffi ile Chrome 120 taklidi yapılarak siteye sızılıyor: {clean_url}")
        # impersonate="chrome120" parametresi ile hedef siteye %100 gerçek bir Chrome gibi el sallıyoruz
        response = curl_requests.get(clean_url, impersonate="chrome120", timeout=15)

        if response.status_code != 200:
            print(f"[Ajan A] Site erişimi reddetti (HTTP {response.status_code}). B1 Planına geçiliyor...")
            return extract_from_url(clean_url)

        # BeautifulSoup ile DOM'u parse et
        soup = BeautifulSoup(response.content, "html.parser")

        # 1. JSON-LD (Schema.org) Arama Motoru Verilerini Çıkar
        json_ld_data = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                json_ld_data.append(data)
            except:
                continue

        # 2. HTML'i Budama (LLM Token Optimizasyonu)
        for element in soup(["script", "style", "nav", "footer", "header", "meta", "link", "svg", "path"]):
            element.extract()
        
        clean_text = soup.get_text(separator=" ", strip=True)
        clean_text = re.sub(r'\s+', ' ', clean_text) # Fazla boşlukları sil
        optimized_text = clean_text[:4000] 

        # LLM İçin Yoğunlaştırılmış Bağlam (Context)
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
        
        print(f"[Ajan A] Veri başarıyla ayıklandı: {extracted_data}")
        return extracted_data

    except Exception as e:
        print(f"[Ajan A] Bağlantı/Ayrıştırma Hatası ({e}). B1 Planına geçiliyor...")
        return extract_from_url(clean_url)

# --- 6. API ENDPOINT'İ ---
@app.post("/analyze")
async def analyze_watch(data: AnalysisRequest):
    print(f"\n--- YENİ SANAL DENEME İSTEĞİ ({data.wristRangeStr} Bilek) ---")
    
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
        watch_features = agentic_universal_scraper(data.productLink)

    kasa_capi = watch_features.get("kasa_capi") if watch_features else None
    
    if not kasa_capi or kasa_capi == "Belirtilmemiş" or kasa_capi is None:
        print("Tüm otonom veri çekme planları başarısız. Kullanıcıya danışılıyor (Plan B2)...")
        return {
            "status": "manual_input_needed",
            "message": "Güvenlik duvarları nedeniyle saatin ölçülerini otomatik okuyamadık. Analiz için lütfen kasa çapını aşağıdan seçin."
        }

    return {
        "status": "success",
        "message": "Model özellikleri başarıyla işlendi.",
        "scraped_data": watch_features
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)