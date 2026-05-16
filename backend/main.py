from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
import re

# Uygulamayı Başlat
app = FastAPI(title="VTO Sanal Deneme API", version="1.0")

# CORS Ayarları (Frontend'in Backend'e sorunsuz istek atabilmesi için zorunludur)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme aşamasında her yere açık, canlıda sadece frontend domaini yazılır
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend'den Gelecek Veri Modeli (JS'deki window.userData ile birebir aynı olmalı)
class AnalysisRequest(BaseModel):
    gender: str
    height: float
    weight: float
    wristPixelDistance: float
    skinColorHex: str
    productLink: str

# --- ARAÇLAR (TOOL CALLING) ---
def scrape_trendyol(url: str):
    print(f"\nHedefe gidiliyor: {url}")
    
    # B PLANIMIZ (Mock Veri): Sistem engellenirse veya hata alırsa kullanılacak yedek veri.
    fallback_data = {
        "kasa_capi": "40mm",
        "materyal": "Çelik",
        "cam": "Safir"
    }

    # Tarayıcı gibi davranmak için sahte kimlik
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # Erişim reddedilirse B Planına geç
        if response.status_code != 200:
            print("Trendyol erişimi reddetti (Anti-Bot). B Planı devreye giriyor!")
            return fallback_data

        soup = BeautifulSoup(response.content, "html.parser")
        extracted_data = {}

        # Trendyol ürün özellikleri listesini bul
        attributes = soup.find_all("li", class_="detail-attr-item")
        
        for attr in attributes:
            text = attr.get_text(strip=True).lower()
            
            # Kasa Çapını Yakala
            if "çap" in text or "kasa çapı" in text:
                match = re.search(r'(\d+)\s*mm', text)
                if match:
                    extracted_data["kasa_capi"] = f"{match.group(1)}mm"
            
            # Materyali Yakala
            elif "kasa materyali" in text or "materyal" in text:
                extracted_data["materyal"] = text.split(":")[-1].strip().title()

        # Eğer veri başarıyla çekildiyse döndür, yoksa B planına geç
        if "kasa_capi" in extracted_data:
            print("Canlı veri başarıyla çekildi!")
            return extracted_data
        else:
            print("Sayfa yüklendi ama kasa çapı bulunamadı. B Planı devreye giriyor!")
            return fallback_data

    except Exception as e:
        print(f"Scraping Hatası: {e}. B Planı devreye giriyor!")
        return fallback_data
    
# --- API ENDPOINTLERİ ---
@app.post("/analyze")
async def analyze_watch(data: AnalysisRequest):
    print("\n--- YENİ ANALİZ İSTEĞİ GELDİ ---")
    print(f"Link: {data.productLink}")
    print(f"Bilek Piksel Mesafesi: {data.wristPixelDistance:.2f} px")
    print(f"Boy: {data.height} cm | Kilo: {data.weight} kg | Ten: {data.skinColorHex}")

    try:
        # Adım 1: Ajan dış dünyaya erişiyor (Canlı ürün verisini çek)
        scraped_data = scrape_trendyol(data.productLink)
        
        # Adım 2: Anatomik Kalibrasyon (1 Pikselin Kaç MM olduğunu bul)
        # Boyun %5.5'ini kullanarak bilek-parmak kökü arasını mm cinsinden tahmin ediyoruz
        estimated_distance_mm = (data.height * 0.055) * 10
        
        # Sıfıra bölünme hatasını (ZeroDivisionError) engellemek için güvenlik kontrolü
        if data.wristPixelDistance > 0:
            mm_per_pixel = estimated_distance_mm / data.wristPixelDistance
            user_wrist_width_mm = round(data.wristPixelDistance * mm_per_pixel)
        else:
            user_wrist_width_mm = 0 # Kamera verisi gelmemişse (Tip 2 manuel senaryo için)

        print(f"Hesaplanan Bilek Genişliği: {user_wrist_width_mm} mm")
        print(f"Çekilen Saat Kasası: {scraped_data.get('kasa_capi')}")
        
        # TODO: Adım 3 - Tüm bu verileri Gemini API'ye gönder
        
        return {
            "status": "success",
            "message": "Saat verileri başarıyla işlendi.",
            "recommendation": f"Sistem bilek genişliğinizi tahmini {user_wrist_width_mm}mm olarak ölçtü. Seçtiğiniz {scraped_data.get('kasa_capi', 'bilinmeyen')} model bileğinizde nasıl duracak hesaplanıyor...",
            "scraped_data": scraped_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)