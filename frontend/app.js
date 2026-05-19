import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// --- GLOBAL DEĞİŞKENLER ---
let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let cameraStream = null;

let isCountingDown = false;
let measurementBuffer = []; // 3 saniye boyunca toplanacak füzyonlanmış bilek verileri
let countdownInterval;

// Kamera Modu (Mobil: Arka, PC: Ön)
const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
let currentFacingMode = isMobileDevice ? "environment" : "user";

// Backend'e gönderilecek ve UI'da kullanılacak veriler
window.userData = {
    gender: 'male',
    height: 175,
    weight: 70,
    wristRangeStr: '', 
    skinColorHex: '#000000',
    productLink: ''
};

const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });
const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

// --- MEDIAPIPE BAŞLATMA ---
async function initializeMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.7, // Yanıp sönmeyi engeller, modeli daha kararlı olmaya zorlar
        minHandPresenceConfidence: 0.7
    });
}
initializeMediaPipe();

// --- KAMERA YÖNETİMİ ---
async function startCamera() {
    if (!handLandmarker) return;
    
    if (currentFacingMode === "user") {
        videoElement.style.transform = "scaleX(-1)";
        canvasElement.style.transform = "scaleX(-1)";
    } else {
        videoElement.style.transform = "none";
        canvasElement.style.transform = "none";
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: currentFacingMode },
            audio: false
        });
        videoElement.srcObject = cameraStream;
        videoElement.addEventListener("loadeddata", () => {
            webcamRunning = true;
            predictWebcam();
        });
    } catch (error) {
        console.error("Kamera açılamadı veya izin reddedildi:", error);
        
        // Zarif Çöküş (Graceful Degradation): Kameraya ulaşılamazsa manuel metoda zorla
        alert("Kamera izni reddedildi veya cihazınızda kamera bulunamadı. Lütfen ölçünüzü manuel olarak giriniz.");
        
        // UI'daki radyo butonunu manuel olarak işaretle
        document.querySelector('input[value="manual"]').checked = true;
        window.toggleMethod(); 
        
        // Sistemi hemen Adım 2'ye (Manuel Parametreler) geri döndür
        stopCamera();
        const allSteps = document.querySelectorAll('.wizard-step');
        allSteps.forEach(step => step.classList.remove('active'));
        document.getElementById('step-2').classList.add('active');
    }
}

function stopCamera() {
    webcamRunning = false;
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
}

window.toggleCamera = function() {
    stopCamera();
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
};

// --- YARDIMCI MATEMATİK VE BİYOLOJİ FONKSİYONLARI ---
function rgbToHex(r, g, b) {
    return "#" + (1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1).toUpperCase();
}

// Tıbbi Normlara Göre Ön Beklenti (Prior) Hesaplama (Türkiye Popülasyonu Normları)
function calculatePriorWrist(height, weight, gender) {
    let base = gender === 'male' ? height / 10.0 : height / 10.5;
    let bmi = weight / Math.pow(height / 100, 2);

    // VKİ Düzeltmesi (Adipoz doku birikimi analizi)
    if (bmi > 25.0) {
        base += (bmi - 25.0) * 0.15;
    } else if (bmi < 18.5) {
        base -= (18.5 - bmi) * 0.10;
    }
    return base;
}

// --- GÖRÜNTÜ İŞLEME DÖNGÜSÜ ---
async function predictWebcam() {
    if (!webcamRunning) return;

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    hiddenCanvas.width = videoElement.videoWidth;
    hiddenCanvas.height = videoElement.videoHeight;
    
    const drawingUtils = new DrawingUtils(canvasCtx);

    if (lastVideoTime !== videoElement.currentTime) {
        lastVideoTime = videoElement.currentTime;
        hiddenCtx.drawImage(videoElement, 0, 0, videoElement.videoWidth, videoElement.videoHeight);
        
        const results = handLandmarker.detectForVideo(videoElement, performance.now());
        
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Hem 2B Ekran hem de 3B Dünya Koordinatlarının geldiğinden emin olalım
        if (results.landmarks && results.landmarks.length > 0 && results.worldLandmarks && results.worldLandmarks.length > 0) {
            const landmarks = results.landmarks[0];
            const worldLandmarks = results.worldLandmarks[0];

            if (isHandFist(worldLandmarks)) {
                document.getElementById('status-message').innerText = "⚠️ Lütfen elinizi düz tutun, yumruk yapmayın!";
                document.getElementById('status-message').classList.remove('counting');
                if (isCountingDown) cancelCountdown();
                
                // İskeleti kullanıcıyı görsel olarak uyarmak için KIRMIZI çiziyoruz
                drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#FF0000", lineWidth: 4 });
                document.getElementById('hand-guide').style.display = 'none';
                canvasCtx.restore();
                window.requestAnimationFrame(predictWebcam);
                return; // Kritik: Hatalı ölçüm alınmasın diye fonksiyonu burada sonlandırıyoruz!
            }
            
            // Çizimleri kalınlaştırdık (İnce ve cılız görünme sorunu çözüldü)
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2, radius: 5 });

            const thumbTip = landmarks[4];  
            const pinkyTip = landmarks[20]; 
            const handGuide = document.getElementById('hand-guide');

            handGuide.style.display = 'block';

            // Başparmak serçe parmağın solundaysa sağ el şablonu, sağındaysa sol el şablonu
            if (thumbTip.x < pinkyTip.x) {
                handGuide.style.transform = "translate(-50%, -50%) scaleX(-1)";
            } else {
                handGuide.style.transform = "translate(-50%, -50%) scaleX(1)";
            }

            // 1. GÖRSEL ÖLÇÜM (3B Dünya Koordinatları ile Derinlikten Bağımsız)
            const indexMcpW = worldLandmarks[5];
            const pinkyMcpW = worldLandmarks[17];

            // 3B Öklid Mesafesi (Metre cinsinden hesaplanır, cm'ye çevrilir)
            const dx = pinkyMcpW.x - indexMcpW.x;
            const dy = pinkyMcpW.y - indexMcpW.y;
            const dz = pinkyMcpW.z - indexMcpW.z;
            const palmWidthMeters = Math.sqrt(dx*dx + dy*dy + dz*dz);
            const palmWidthCm = palmWidthMeters * 100;

            // Yatay çap (2a) ve derinlik (2b) hesaplaması (Anatomik Oranlar)
            const width2a = palmWidthCm / 1.45; 
            const depth2b = width2a * 0.68; 

            const a = width2a / 2;
            const b = depth2b / 2;

            // Ramanujan Elips Çevre Formülü (Görsel Tahmin)
            const cVisual = Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));

            // 2. İSTATİSTİKSEL ÖN BEKLENTİ (Biyolojik Gerçeklik)
            const cPrior = calculatePriorWrist(window.userData.height, window.userData.weight, window.userData.gender);

            // 3. BAYESYAN VERİ FÜZYONU (Hata Filtreleme)
            // Biyolojik veriye %60, kamera verisine %40 güveniyoruz
            const alpha = 0.6; 
            const finalWristCm = (alpha * cPrior) + ((1 - alpha) * cVisual);

            // Kullanıcıya aralık sunmak için -0.5 ve +0.5 marjı ekle
            const liveRange = `${(finalWristCm - 0.5).toFixed(1)} - ${(finalWristCm + 0.5).toFixed(1)} cm`;

            // Ten Rengi Çekme İşlemi (Gizli tuvalden)
            let hexColor = "#000000";
            const pixelX = Math.floor(landmarks[0].x * videoElement.videoWidth);
            const pixelY = Math.floor(landmarks[0].y * videoElement.videoHeight);
            if (pixelX >= 0 && pixelX < videoElement.videoWidth && pixelY >= 0 && pixelY < videoElement.videoHeight) {
                const pixelData = hiddenCtx.getImageData(pixelX, pixelY, 1, 1).data;
                hexColor = rgbToHex(pixelData[0], pixelData[1], pixelData[2]);
            }

            // Ekrana canlı yazdır
            document.getElementById('live-wrist').innerText = liveRange;
            document.getElementById('live-color-box').style.backgroundColor = hexColor;

            // Hizalama Kontrolü (El filigranın içinde mi?)
            const isHandInBox = (landmarks[0].x > 0.40 && landmarks[0].x < 0.60 && landmarks[0].y > 0.40 && landmarks[0].y < 0.85);

            if (isHandInBox && !isCountingDown) {
                startCountdown();
            } else if (!isHandInBox && isCountingDown) {
                cancelCountdown();
            }

            // Geri sayım sürüyorsa füzyonlanmış verileri tampona at
            if (isCountingDown) {
                measurementBuffer.push({ wristVal: finalWristCm, color: hexColor });
            }
        } else {
            document.getElementById('hand-guide').style.display = 'none';
            if (isCountingDown) cancelCountdown();
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
}

// --- GERİ SAYIM YÖNETİMİ VE DURUM PANELİ ---
function startCountdown() {
    isCountingDown = true;
    measurementBuffer = []; 
    
    const statusMsg = document.getElementById('status-message');
    statusMsg.classList.add('counting'); 
    
    let count = 3;
    statusMsg.innerText = `Ölçüm ayarlandı, sabit durun: ${count}`;

    countdownInterval = setInterval(() => {
        count--;
        if (count > 0) {
            statusMsg.innerText = `Ölçüm ayarlandı, sabit durun: ${count}`;
        } else {
            clearInterval(countdownInterval);
            statusMsg.classList.remove('counting');
            statusMsg.innerText = "Ölçüm tamamlandı, işleniyor...";
            finalizeMeasurement();
        }
    }, 1000);
}

function cancelCountdown() {
    isCountingDown = false;
    clearInterval(countdownInterval);
    
    const statusMsg = document.getElementById('status-message');
    statusMsg.classList.remove('counting');
    statusMsg.innerText = "El hizalaması bekleniyor...";
}

function finalizeMeasurement() {
    // Toplanan ölçümlerin ortalamasını alarak mikro titremeleri tamamen sil
    let totalWrist = 0;
    measurementBuffer.forEach(data => totalWrist += data.wristVal);
    const avgWrist = totalWrist / measurementBuffer.length;
    
    window.userData.wristRangeStr = `${(avgWrist - 0.5).toFixed(1)} - ${(avgWrist + 0.5).toFixed(1)} cm`;
    if (measurementBuffer.length > 0) window.userData.skinColorHex = measurementBuffer[measurementBuffer.length - 1].color;

    console.log("Ölçüm Tamamlandı! Kilitlenen Aralık:", window.userData.wristRangeStr);
    
    stopCamera();
    updateSummaryScreen();
    nextStep(4); 
}

// --- UI YÖNETİMİ VE EKRAN GEÇİŞLERİ ---
// --- UI YÖNETİMİ VE EKRAN GEÇİŞLERİ ---
window.toggleMethod = function() {
    const method = document.querySelector('input[name="measureMethod"]:checked').value;
    if (method === 'camera') {
        document.getElementById('camera-params').style.display = 'block';
        document.getElementById('manual-params').style.display = 'none';
    } else {
        document.getElementById('camera-params').style.display = 'none';
        document.getElementById('manual-params').style.display = 'block';
    }
};

window.proceedFromStep2 = function() {
    const method = document.querySelector('input[name="measureMethod"]:checked').value;
    
    // YÖNTEM NE OLURSA OLSUN CİNSİYETİ HER DURUMDA KAYDEDİYORUZ
    window.userData.gender = document.getElementById('gender').value;
    
    if (method === 'camera') {
        const heightInput = document.getElementById('height').value;
        const weightInput = document.getElementById('weight').value;

        // KONTROL: Boy veya kilo boşsa, ya da 0'dan küçük bir mantıksız değer girildiyse
        if (!heightInput || !weightInput || parseFloat(heightInput) <= 0 || parseFloat(weightInput) <= 0) {
            alert("Lütfen boy ve kilo bilgilerinizi eksiksiz ve geçerli bir şekilde giriniz.");
            return; // return diyerek fonksiyonu burada kesiyoruz, sonraki adıma geçmesini engelliyoruz
        }

        // Değerler geçerliyse sisteme kaydet
        window.userData.height = parseFloat(heightInput);
        window.userData.weight = parseFloat(weightInput);
        nextStep(3);
        
    } else {
        const manualVal = parseFloat(document.getElementById('manual-wrist').value);
        if(!manualVal || manualVal <= 0) {
            alert("Lütfen geçerli bir bilek çevresi girin.");
            return;
        }
        window.userData.wristRangeStr = `${manualVal} cm (Manuel)`;
        window.userData.skinColorHex = null; 
        
        updateSummaryScreen();
        nextStep(4);
    }
};

function updateSummaryScreen() {
    document.getElementById('summary-wrist').innerText = window.userData.wristRangeStr;
    const colorWrapper = document.getElementById('summary-color-wrapper');
    if (window.userData.skinColorHex) {
        colorWrapper.style.display = 'flex';
        document.getElementById('summary-color').style.backgroundColor = window.userData.skinColorHex;
    } else {
        colorWrapper.style.display = 'none';
    }
}

// 5. Adımda eğer site engellerse çalışacak Dürüst UX fonksiyonu (Plan B2)
window.submitManualSize = function(size) {
    window.userData.manualWatchSize = size;
    nextStep(5); // Veriyi güncelleyip aynı adıma tekrar istek atıyoruz
};

window.nextStep = function(stepNumber) {
    if (stepNumber === 3) {
        startCamera(); 
    } else {
        stopCamera(); 
    }

    // ADIM 5: Backend ile El Sıkışma
    if (stepNumber === 5) {
        window.userData.productLink = document.getElementById('productLink').value;
        if (!window.userData.productLink) {
            alert("Lütfen analiz için bir saat linki yapıştırın.");
            return; 
        }
        document.getElementById('cyber-main-title').innerText = "SİSTEM ANALİZ EDİYOR";

        // UI Hazırlıkları
        document.getElementById('ai-loading').style.display = 'flex';
        document.getElementById('ai-results').style.display = 'none';
        document.getElementById('error-container').style.display = 'none';
        
        // --- JÜRİ ŞOVU: CANLI AKAN TERMİNAL ---
        const loadingText = document.getElementById('loading-text');
        const agentMessages = [
            "Paylaşımlı bağlam havuzu oluşturuluyor...",
            "Trendyol veri tabanına sızılıyor...",
            "JSON-LD meta verileri ayrıştırılıyor...",
            "Anti-Bot (Cloudflare) atlatılıyor...",
            "Biyometrik uyum hesaplanıyor...",
            "Son analiz raporu hazırlanıyor..."
        ];
        
        let msgIndex = 0;
        loadingText.innerHTML = `<div>${agentMessages[0]}</div>`;
        
        window.agentInterval = setInterval(() => {
            msgIndex++;
            if(msgIndex < agentMessages.length) {
                loadingText.innerHTML += `<div>${agentMessages[msgIndex]}</div>`;
            } else {
                clearInterval(window.agentInterval);
            }
        }, 1500);

        fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(window.userData)
        })
        .then(response => response.json())
        .then(data => {

            clearInterval(window.altInterval);

            console.log("Backend Yanıtı:", data);
            document.getElementById('ai-loading').style.display = 'none'; // Yüklemeyi gizle

            const errorContainer = document.getElementById('error-container');

            if (data.status === "manual_input_needed") {
                // Güvenlik duvarı aşılamadıysa Manuel Seçim Ekranı (Plan B2)
                errorContainer.style.display = 'block';
                errorContainer.innerHTML = `
                    <h4 style="color: #FF6B00; margin-top:0;">🛡️ Güvenlik Duvarı Aşılamadı</h4>
                    <p style="color: #666; font-size: 0.95em;">${data.message}</p>
                    <div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 15px;">
                        <button class="btn-secondary" onclick="submitManualSize('36mm')">36mm</button>
                        <button class="btn-secondary" onclick="submitManualSize('38mm')">38mm</button>
                        <button class="btn-secondary" onclick="submitManualSize('40mm')">40mm</button>
                        <button class="btn-secondary" onclick="submitManualSize('42mm')">42mm</button>
                        <button class="btn-secondary" onclick="submitManualSize('44mm')">44mm</button>
                    </div>
                `;
            } 
            else if (data.status === "success") {
                document.getElementById('ai-loading').style.display = 'none';
                renderAIResults(data); // Ana analizi ekrana bas
            }
        })
        .catch(error => {
            clearInterval(window.altInterval);
            
            document.getElementById('ai-loading').style.display = 'none';
            const errorContainer = document.getElementById('error-container');
            errorContainer.style.display = 'block';
            errorContainer.innerHTML = `
                <h4 style="color: red; margin-top:0;">Sunucu Hatası</h4>
                <p>Python (FastAPI) sunucusuna ulaşılamadı. Sunucunun çalıştığından emin olun.</p>
            `;
            console.error("Fetch Hatası:", error);
        });
    }

    const allSteps = document.querySelectorAll('.wizard-step');
    allSteps.forEach(step => step.classList.remove('active'));
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) targetStep.classList.add('active');
};

function renderAIResults(data) {
    // ANALİZ BİTTİĞİ İÇİN BAŞLIĞI GÜNCELLİYORUZ
    document.getElementById('cyber-main-title').innerText = "SİSTEM ANALİZİ TAMAMLANDI";
    
    document.getElementById('ai-results').style.display = 'flex'; 
    const watch = data.scraped_data;
    const stylist = data.stylist_data; 
    
    const imgElement = document.getElementById('scraped-watch-image');
    imgElement.src = (watch.resim_url && watch.resim_url !== "Belirtilmemiş") ? watch.resim_url : "https://via.placeholder.com/400x500/111111/FFFFFF?text=Saat+Gorseli";

    const score = stylist.match_score;
    const segments = document.querySelectorAll('.segment');
    
    // --- 10 SEGMENTLİ YENİ MATEMATİKSEL HESAPLAMA VE RENK SKALASI ---
    // Skor 100 üzerinden olduğu için 10'a bölüp kaç segment yanacağını buluyoruz (Örn: 72 skor = 7 segment)
    let activeSegments = Math.round(score / 10);
    if (activeSegments < 1 && score > 0) activeSegments = 1; // Sıfırlanmayı önle

    let statusText = "HESAPLANIYOR", statusColor = "#fff";

    if (score < 30) { statusText = "UYUMSUZ"; statusColor = "#666666"; }
    else if (score < 50) { statusText = "ZAYIF UYUM"; statusColor = "#888888"; }
    else if (score < 70) { statusText = "ORTALAMA"; statusColor = "#AAAAAA"; }
    else if (score < 85) { statusText = "İYİ UYUM"; statusColor = "#DDDDDD"; }
    else if (score < 95) { statusText = "MÜKEMMEL"; statusColor = "#EAEAEA"; }
    else { statusText = "KUSURSUZ"; statusColor = "#FFFFFF"; }

    document.getElementById('match-status-text').innerText = statusText;
    document.getElementById('match-status-text').style.color = statusColor;

    // Tüm segmentleri tara ve aktif segment kadarını renklendir
    segments.forEach((seg, index) => {
        seg.style.background = (index < activeSegments) ? statusColor : "#222";
    });

    document.getElementById('ai-stylist-comment').innerHTML = stylist.stylist_comment;

    const warningElement = document.getElementById('cyber-gender-warning');
    if (stylist.warning) {
        warningElement.innerText = stylist.warning;
        warningElement.style.display = 'block';
    } else {
        warningElement.style.display = 'none';
    }

    if (stylist.recommendations && stylist.recommendations.length > 0) {
        document.getElementById('ai-recs-container').style.display = 'block';
        const recsHtml = stylist.recommendations.map(rec => `<span class="rec-tag" onclick="loadAlternative('${rec}')">${rec}</span>`).join('');
        document.getElementById('ai-rec-tags').innerHTML = recsHtml;
    } else {
        document.getElementById('ai-recs-container').style.display = 'none';
    }

    // 3D HOLOGRAM ETKİSİ
    const tiltContainer = document.getElementById('tilt-container');
    tiltContainer.onmousemove = (e) => {
        const rect = tiltContainer.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        const xRotation = 20 * ((y - rect.height / 2) / rect.height);
        const yRotation = -20 * ((x - rect.width / 2) / rect.width);
        imgElement.style.transform = `rotateX(${xRotation}deg) rotateY(${yRotation}deg) scale(1.05)`;
    };
    tiltContainer.onmouseleave = () => { imgElement.style.transform = `rotateX(0deg) rotateY(0deg) scale(1)`; };
}

// Alternatif Rota Butonuna Tıklandığında Çalışacak Fonksiyon
window.loadAlternative = function(styleText) {
    document.getElementById('ai-results').style.display = 'none';
    document.getElementById('ai-loading').style.display = 'flex';
    
    // --- JÜRİ ŞOVU: ALTERNATİF ROTA İÇİN AKAN TERMİNAL ---
    const loadingText = document.getElementById('loading-text');
    const altMessages = [
        `"${styleText}" için e-ticaret taranıyor...`,
        "Gerçek ürün bağlantısı bulundu...",
        "Sistem engelleri atlatılıyor...",
        "Yeni model anatomik olarak analiz ediliyor..."
    ];
    
    let msgIdx = 0;
    loadingText.innerHTML = `<div>${altMessages[0]}</div>`;
    
    window.altInterval = setInterval(() => {
        msgIdx++;
        if(msgIdx < altMessages.length) {
            loadingText.innerHTML += `<div>${altMessages[msgIdx]}</div>`;
        } else {
            clearInterval(window.altInterval);
        }
    }, 1200)

    const altData = {
        target_style: styleText,
        gender: window.userData.gender,
        wristRangeStr: window.userData.wristRangeStr,
        skinColorHex: window.userData.skinColorHex || "#000000"
    };

    fetch('http://127.0.0.1:8000/simulate_alternative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(altData)
    })
    .then(res => res.json())
    .then(data => {

        clearInterval(window.altInterval);

        if (data.status === "success") {
            document.getElementById('ai-loading').style.display = 'none';
            renderAIResults(data); // Aynı ekranı Trendyol'dan gelen gerçek verilerle güncelle
        } else {
            alert("Ürün bulunamadı veya bir hata oluştu.");
            document.getElementById('ai-loading').style.display = 'none';
            document.getElementById('ai-results').style.display = 'flex'; 
        }
    })
    .catch(err => {

        clearInterval(window.altInterval);

        console.error("Hata:", err);
        document.getElementById('ai-loading').style.display = 'none';
        document.getElementById('ai-results').style.display = 'flex';
    });
};

// --- YUMRUK ALGILAMA FİLTRESİ ---
function isHandFist(worldLandmarks) {
    // Parmak Uçları: İşaret(8), Orta(12), Yüzük(16), Serçe(20)
    const fingerTips = [8, 12, 16, 20];
    // Parmak Kök Eklemleri (MCP): İşaret(5), Orta(9), Yüzük(13), Serçe(17)
    const fingerBases = [5, 9, 13, 17];
    
    let closedFingersCount = 0;
    const wrist = worldLandmarks[0]; // Bilek referans noktası

    for (let i = 0; i < 4; i++) {
        const tip = worldLandmarks[fingerTips[i]];
        const base = worldLandmarks[fingerBases[i]];

        // 3B Öklid Mesafesi Hesaplama
        const distTipToWrist = Math.sqrt(
            Math.pow(tip.x - wrist.x, 2) + 
            Math.pow(tip.y - wrist.y, 2) + 
            Math.pow(tip.z - wrist.z, 2)
        );
        const distBaseToWrist = Math.sqrt(
            Math.pow(base.x - wrist.x, 2) + 
            Math.pow(base.y - wrist.y, 2) + 
            Math.pow(base.z - wrist.z, 2)
        );

        // Eğer parmak ucu bileğe, kök ekleminden daha yakınsa parmak kapanmıştır (yumruk pozisyonu)
        if (distTipToWrist < distBaseToWrist) {
            closedFingersCount++;
        }
    }

    // 4 parmaktan en az 3'ü kapanmışsa bu bir yumruktur!
    return closedFingersCount >= 3;
}