# ⌚ VTO — Sanal Saat Deneme Asistanı

> **"Saatini denemeden alma."**  
> Yapay zeka ve bilgisayarlı görü destekli, gerçek zamanlı bilek ölçüm ve saat uyum analizi sistemi.

---

## 📌 Projeye Genel Bakış

VTO (**Virtual Try-On**), kullanıcıların bir e-ticaret sitesindeki herhangi bir saati satın almadan önce bileğine ne kadar uyacağını tahmin eden, uçtan uca çalışan bir yapay zeka uygulamasıdır.

Sistem üç temel işi yapar:

1. **Bilek Ölçümü** — Kamera veya manuel giriş yoluyla kullanıcının bilek çevresini hesaplar.
2. **Ürün Veri Çekme** — Yapıştırılan e-ticaret linkinden saatin fiziksel özelliklerini (kasa çapı, materyal, renk) otomatik olarak toplar.
3. **Stilist AI Analizi** — Biyometrik veriler ile ürün verilerini birleştirerek kişiselleştirilmiş bir uyum skoru ve yorum üretir.

---

## 🏗️ Mimari ve Teknoloji Yığını

```
┌─────────────────────────────────────────┐
│             FRONTEND (Tarayıcı)         │
│                                         │
│  index.html  ──  style.css              │
│       │                                 │
│    app.js (ES Module)                   │
│       │                                 │
│  MediaPipe HandLandmarker (WASM/GPU)    │
│  ├── El tespiti & landmark tracking     │
│  ├── 3B dünya koordinatları             │
│  ├── Bayesyan veri füzyonu              │
│  └── Ten rengi tespiti (canvas pixel)   │
└──────────────────┬──────────────────────┘
                   │ HTTP POST (JSON)
                   ▼
┌─────────────────────────────────────────┐
│             BACKEND (Python)            │
│                                         │
│  FastAPI  ──  Uvicorn                   │
│       │                                 │
│  /analyze  endpoint                     │
│  ├── Kazıma Ajanı (curl_cffi + BS4)     │
│  ├── Gemini AI Ajanı (veri çekme)       │
│  └── Stilist Ajan (uyum analizi)        │
│                                         │
│  /simulate_alternative  endpoint        │
│  ├── Trendyol Arama Ajanı              │
│  └── Gemini AI Ajanı (öneri analizi)   │
└─────────────────────────────────────────┘
```

### Kullanılan Teknolojiler

| Katman | Teknoloji | Amaç |
|---|---|---|
| Frontend | Vanilla JS (ES Modules) | Uygulama mantığı |
| Görüntü İşleme | MediaPipe HandLandmarker | El/bilek tespiti |
| Backend | FastAPI + Uvicorn | REST API sunucusu |
| Web Kazıma | curl_cffi + BeautifulSoup4 | Anti-bot atlatma, HTML parse |
| AI (Scraping) | Google Gemini AI | Ürün verisini HTML'den çıkarma |
| AI (Analiz) | Google Gemini AI | Stilist yorum ve uyum skoru |
| Veri Modeli | Pydantic | İstek/yanıt doğrulama |
| CSS | Glassmorphism + Custom | Lüks minimalist tasarım |

---

## 📂 Proje Dosya Yapısı

```
vto/
├── index.html          # Sihirbaz adım UI'ı (5 adım)
├── style.css           # Glassmorphism + futuristic dashboard stilleri
├── app.js              # Frontend mantığı (MediaPipe, kamera, fetch)
├── main.py             # FastAPI backend (ajan mimarisi)
├── requirements.txt    # Python bağımlılıkları
└── .env                # API anahtarları (git'e eklenmemeli!)
```

---

## 🔄 Kullanıcı Akışı (5 Adım)

```
[Adım 1] Karşılama Ekranı
    └─► Kullanıcı "Hemen Başla" butonuna tıklar

[Adım 2] Ölçüm Yöntemi Seçimi
    ├─► Kamera ile Akıllı Ölçüm (Önerilen)
    │       └─► Boy + Kilo + Cinsiyet girişi
    └─► Manuel Giriş
            └─► Bilek çevresi (cm) girişi

[Adım 3] Kamera ile Bilek Analizi  ← (Sadece kamera yolu seçildiyse)
    ├─► MediaPipe ile gerçek zamanlı el tespiti
    ├─► Yumruk algılama filtresi (hatalı ölçümü önler)
    ├─► 3 saniyelik geri sayımlı ölçüm kilitleme
    └─► Bayesyan füzyonu ile nihai bilek değeri hesaplama

[Adım 4] Özet & Ürün Linki
    ├─► Hesaplanan bilek çevresi ve ten rengi gösterimi
    └─► E-ticaret ürün linki girişi (Trendyol, Amazon vb.)

[Adım 5] AI Analiz Panosu
    ├─► Ürün verisi otomatik çekilir
    ├─► Ergonomik uyum skoru (0–100) hesaplanır
    ├─► Stilist yorum ve uyarılar gösterilir
    └─► Alternatif ürün önerileri (Trendyol'da canlı arama)
```

---

## 🧠 Teknik Derinlik: Bilek Ölçüm Algoritması

Sistem, birden fazla veri kaynağını birleştiren **Bayesyan Veri Füzyonu** kullanır.

### 1. Görsel Ölçüm (MediaPipe 3B Koordinatları)

MediaPipe'ın `worldLandmarks` verisi (metre cinsinden gerçek dünya koordinatları) kullanılarak avuç genişliği ölçülür:

```
indexMCP (nokta 5)  ←→  pinkyMCP (nokta 17)
       palmWidthCm = 3B Öklid Mesafesi × 100
```

Ardından bileğin elips şeklinde olduğu varsayılarak **Ramanujan Elips Çevre Formülü** uygulanır:

```
width_2a = palmWidthCm / 1.45   (Anatomik oran)
depth_2b = width_2a × 0.68      (Anatomik oran)

a = width_2a / 2
b = depth_2b / 2

cVisual = π × (3(a+b) − √((3a+b)(a+3b)))
```

### 2. İstatistiksel Ön Beklenti (Biyolojik Prior)

Kullanıcının boy, kilo ve cinsiyetine dayalı **Türkiye popülasyonu normlarına** göre hesaplanan beklenen bilek değeri:

```python
def calculatePriorWrist(height, weight, gender):
    base = height / 10.0   # Erkek
    # veya
    base = height / 10.5   # Kadın

    bmi = weight / (height / 100) ** 2

    # VKİ düzeltmesi (Adipoz doku analizi)
    if bmi > 25.0:
        base += (bmi - 25.0) * 0.15
    elif bmi < 18.5:
        base -= (18.5 - bmi) * 0.10

    return base
```

### 3. Bayesyan Füzyon

Biyolojik veriye %60, kamera verisine %40 ağırlık verilerek nihai değer hesaplanır:

```
finalWrist = 0.6 × cPrior + 0.4 × cVisual
```

Sonuç kullanıcıya ±0.5 cm aralığı olarak sunulur: `"16.2 – 17.2 cm"`

### 4. Ten Rengi Tespiti

Bileğin kök noktasına (landmark[0]) karşılık gelen piksel, gizli bir `<canvas>` üzerinden okunur ve HEX renk koduna çevrilir. Bu veri, stilist analizde saat materyali ile ten uyumunu değerlendirmek için kullanılır.

### 5. Yumruk Algılama Filtresi

Yanlış pozisyonlarda ölçüm alınmasını önlemek için gerçek zamanlı yumruk tespiti yapılır:

```
Her parmak için:
  distTipToWrist < distBaseToWrist → parmak kapalı

4 parmaktan 3'ü kapalıysa → YUMRUK → ölçüm iptal
```

---

## 🤖 Backend Ajan Mimarisi

Backend, **Paylaşımlı Bağlam (Shared Context)** mimarisi üzerine kurulmuş birden fazla otonom ajandan oluşur.

```python
shared_context = {
    "user_anatomy": { cinsiyet, bilek, ten_rengi, ... },
    "watch_data":   None,  # Kazıma Ajanı doldurur
    "stylist_report": None, # Stilist Ajan doldurur
    "system_status": "initializing"
}
```

### Ajan A: Evrensel Web Kazıyıcı

**Görev:** Herhangi bir e-ticaret URL'sinden saat verisi çekmek.

**Yöntem:**
1. `curl_cffi` ile `Chrome 120` taklidi yaparak siteye bağlanır (Cloudflare/bot koruması atlatılır).
2. BeautifulSoup ile HTML parse edilir, `<script type="application/ld+json">` (Schema.org) verileri çıkarılır.
3. Temizlenmiş HTML metni + JSON-LD verisi Gemini AI'a gönderilir.
4. Gemini, saatin `kasa_capi`, `materyal`, `renk`, `kordon`, `resim_url` bilgilerini JSON formatında döndürür.

**Fallback Planları:**
- **Plan B1:** Site erişimi engellendiyse URL içindeki `(\d{2})mm` pattern'i ile kasa çapı aranır.
- **Plan B2:** Hiçbir otonom yöntem işe yaramazsa frontend'de kullanıcıya manuel seçim ekranı açılır (36mm, 38mm, 40mm, 42mm, 44mm).

### Ajan B: Stilist AI

**Görev:** Kullanıcı biyometrisi + saat özellikleri → Uyum skoru ve kişiselleştirilmiş yorum.

Gemini'ye gönderilen prompt, ajanın şu çıktıları üretmesini zorunlu kılar:

```json
{
  "match_score": 78,
  "warning": "Bu saat erkek saatidir.",
  "stylist_comment": "<HTML rozet destekli analiz metni>",
  "recommendations": ["38mm çelik klasik", "40mm siyah deri"]
}
```

**Skor → Durum eşlemesi:**

| Skor | Durum | Renk |
|---|---|---|
| 0–29 | UYUMSUZ | `#666666` |
| 30–49 | ZAYIF UYUM | `#888888` |
| 50–69 | ORTALAMA | `#AAAAAA` |
| 70–84 | İYİ UYUM | `#DDDDDD` |
| 85–94 | MÜKEMMEL | `#EAEAEA` |
| 95–100 | KUSURSUZ | `#FFFFFF` |

**İnceleme Ajanı (Guardrail):** Stilist ajanın çıktısını doğrular. JSON içinde çift tırnak tespit edilirse otomatik düzeltir; `match_score` eksikse 70 olarak atar.

### Arama Ajanı: Alternatif Trendyol Bulucu

**Görev:** Kullanıcı bir öneri etiketine tıkladığında Trendyol'da gerçek ürün bulmak.

**Yöntem (Dar'dan Genişe Arama Stratejisi):**

```python
terms = [
    "39mm çelik spor saat",  # 3 kelime
    "39mm çelik saat",       # 2 kelime
    "çelik saat"             # 1 kelime (Kesin bulur)
]
```

Her arama sonucunda `-p-\d+` regex pattern'i ile Trendyol ürün linkini avlar.

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.9+
- Google Gemini API anahtarı

### 1. Backend Kurulumu

```bash
# Projeyi klonlayın
git clone <repo-url>
cd vto

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Ortam Değişkenleri

Proje kökünde `.env` dosyası oluşturun:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ `.env` dosyasını kesinlikle git'e eklemeyin. `.gitignore`'a ekleyin.

### 3. Backend'i Başlatın

```bash
python main.py
# veya
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API dokümantasyonu: `http://localhost:8000/docs`

### 4. Frontend'i Açın

`index.html` dosyasını bir yerel HTTP sunucusu üzerinden açın (ES modülleri doğrudan `file://` ile çalışmaz):

```bash
# Python ile basit sunucu
python -m http.server 5500

# Node.js ile
npx serve .
```

Tarayıcıda `http://localhost:5500` adresini açın.

---

## 🔌 API Referansı

### `POST /analyze`

Ana analiz endpoint'i.

**İstek Gövdesi:**

```json
{
  "gender": "male",
  "height": 178,
  "weight": 75,
  "wristRangeStr": "16.5 - 17.5 cm",
  "skinColorHex": "#C68642",
  "productLink": "https://www.trendyol.com/...",
  "manualWatchSize": null
}
```

**Başarılı Yanıt:**

```json
{
  "status": "success",
  "scraped_data": {
    "kasa_capi": "42mm",
    "materyal": "Çelik",
    "renk": "Siyah",
    "kordon": "Deri",
    "resim_url": "https://...",
    "kaynak": "curl_cffi + JSON-LD + Gemini AI"
  },
  "stylist_data": {
    "match_score": 82,
    "warning": null,
    "stylist_comment": "<HTML yorum>",
    "recommendations": ["40mm çelik klasik", "42mm siyah deri"]
  }
}
```

**Fallback Yanıtı (Güvenlik Duvarı):**

```json
{
  "status": "manual_input_needed",
  "message": "Güvenlik duvarları nedeniyle saatin ölçülerini otomatik okuyamadık..."
}
```

---

### `POST /simulate_alternative`

Kullanıcı öneri etiketine tıkladığında Trendyol'da gerçek ürün bulup analiz eder.

**İstek Gövdesi:**

```json
{
  "target_style": "38mm çelik klasik",
  "gender": "male",
  "wristRangeStr": "16.5 - 17.5 cm",
  "skinColorHex": "#C68642"
}
```

**Yanıt:** `/analyze` ile aynı yapıda `success` yanıtı.

---

## ⚙️ Yapılandırma

`main.py` içindeki `DEMO_MODE` flag'i geliştirme sırasında gerçek web istekleri yapmadan çalışmanızı sağlar:

```python
DEMO_MODE = True  # Gerçek site yerine yerel katalog kullanılır
```

Demo katalog, `amazon.com`, `trendyol.com` ve `ty.gl` domainleri için örnek veri içerir.

---

## 📱 Cihaz Uyumluluğu

| Özellik | Masaüstü | Mobil |
|---|---|---|
| Varsayılan Kamera | Ön kamera | Arka kamera |
| Kamera Çevirme | Buton gizli | Buton görünür |
| Canvas Ayna Efekti | Aktif (ön için) | Pasif (arka için) |
| Viewport | Tam genişlik | Responsive |

Kamera izni reddedilirse sistem otomatik olarak manuel ölçüm moduna geçer ve kullanıcıyı bilgilendirir.

---

## 🔒 Güvenlik Notları

- CORS tüm originlere açıktır (`allow_origins=["*"]`). Production'da kısıtlanmalıdır.
- Gemini API anahtarı yalnızca backend'de tutulur, frontend'e hiçbir şekilde iletilmez.
- `curl_cffi` ile yapılan Chrome taklidi yalnızca kamuya açık e-ticaret sayfaları için kullanılmaktadır.

---

## 🗺️ Gelecek Geliştirmeler

- [ ] Trendyol dışı platformlar için genişletilmiş arama ajanı
- [ ] Saat görseli üzerine bileğe bindirme (AR overlay)
- [ ] Kullanıcı geçmiş ölçümleri için yerel depolama
- [ ] Çoklu saat karşılaştırma modu
- [ ] Ten rengi paleti ile materyal uyum görselleştirmesi

---

## 📄 Lisans

Bu proje bir hackathon kapsamında geliştirilmiştir.
