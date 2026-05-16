from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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

# Ana İstek Noktası (Endpoint)
@app.post("/analyze")
async def analyze_watch(data: AnalysisRequest):
    print("--- YENİ ANALİZ İSTEĞİ GELDİ ---")
    print(f"Kullanıcı Linki: {data.productLink}")
    print(f"Bilek Piksel Mesafesi: {data.wristPixelDistance}")
    print(f"Boy: {data.height} cm | Ten Rengi: {data.skinColorHex}")

    try:
        # TODO: Adım 1 - Trendyol Linkinden "Kasa Çapı" ve "Materyal" verilerini kazı (Scraping)
        # scraper_data = scrape_trendyol(data.productLink)
        
        # TODO: Adım 2 - Piksel/Boy oranını kullanarak gerçek bilek milimetresini hesapla
        # user_wrist_mm = calculate_wrist_mm(data.height, data.wristPixelDistance)
        
        # TODO: Adım 3 - Tüm bu verileri Gemini API'ye gönder ve otonom yorum al
        # gemini_response = ask_gemini(user_wrist_mm, scraper_data, data.skinColorHex)

        # Şimdilik Frontend'i test etmek için sahte (mock) bir yanıt dönüyoruz
        return {
            "status": "success",
            "message": "Saat verileri başarıyla çekildi. Bilek yapınıza (Tahmini 16cm) ve ten renginize uygunluğu analiz ediliyor.",
            "recommendation": "Seçtiğiniz 42mm model bileğinizde agresif durabilir, 40mm alternatifleri denemek ister misiniz?",
            "scraped_data": {"kasa_capi": "42mm", "materyal": "Çelik"}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Sunucuyu 8000 portunda çalıştır
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)