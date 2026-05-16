import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// --- GLOBAL DEĞİŞKENLER VE VERİ DEPOSU ---
let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let cameraStream = null;
let isCountingDown = false;
let measurementBuffer = []; // 3 saniye boyunca alınan ölçümlerin ortalamasını almak için

// Backend'e gönderilecek kullanıcı verileri
window.userData = {
    gender: '',
    height: 175,
    weight: 70,
    wristPixelDistance: 0,
    skinColorHex: '#000000',
    productLink: ''
};

// DOM Elementleri
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });

// Gizli canvas (Ten rengi okumak için)
const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

// Varsayılan olarak mobil ise arka kamera, masaüstü ise ön kamera
const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
let currentFacingMode = isMobileDevice ? "environment" : "user";

// --- MEDIAPIPE BAŞLATMA ---
async function initializeMediaPipe() {
    console.log("Yapay Zeka Modelleri Yükleniyor...");
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("Modeller Hazır!");
}
initializeMediaPipe();

// --- KAMERA YÖNETİMİ ---
async function startCamera() {
    if (!handLandmarker) {
        alert("Modeller henüz yüklenmedi, lütfen bekleyin.");
        return;
    }

    // Ayna efekti (Sadece ön kameradaysa aynala)
    if (currentFacingMode === "user") {
        videoElement.style.transform = "scaleX(-1)";
        canvasElement.style.transform = "scaleX(-1)";
    } else {
        videoElement.style.transform = "none";
        canvasElement.style.transform = "none";
    }

    const constraints = {
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: currentFacingMode },
        audio: false
    };

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = cameraStream;
        videoElement.addEventListener("loadeddata", () => {
            webcamRunning = true;
            predictWebcam();
        });
    } catch (error) {
        console.error("Kamera açılamadı:", error);
    }
}

// YENİ: Kamera Çevirme Fonksiyonu
window.toggleCamera = function() {
    // Mevcut akışı durdur
    stopCamera();
    
    // Modu değiştir
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    
    // Yeniden başlat
    startCamera();
};

function stopCamera() {
    webcamRunning = false;
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
}

// --- GÖRÜNTÜ İŞLEME DÖNGÜSÜ ---
function rgbToHex(r, g, b) {
    return "#" + (1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1).toUpperCase();
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

        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            
            // ALIŞTIĞIN RENKLERE GERİ DÖNDÜK (Yeşil çizgiler, Kırmızı noktalar)
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 1, radius: 3 });

            const wrist = landmarks[0];
            const indexMcp = landmarks[5];

            // 1. CANLI ÖLÇÜM HESAPLAMALARI
            const x1 = wrist.x * videoElement.videoWidth;
            const y1 = wrist.y * videoElement.videoHeight;
            const x2 = indexMcp.x * videoElement.videoWidth;
            const y2 = indexMcp.y * videoElement.videoHeight;

            const pixelDistance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
            const userHeight = window.userData.height || 175;
            const estimatedCmDistance = userHeight * 0.055; 
            const cmPerPixel = estimatedCmDistance / pixelDistance;
            
            // Tahmini bilek çevresini (cm) ekranda göstermek için basit bir katsayı çarpımı
            const liveWristCm = (pixelDistance * cmPerPixel * 3.5).toFixed(1); 

            // 2. TEN RENGİ TESPİTİ
            let hexColor = "#000000";
            const pixelX = Math.floor(x1);
            const pixelY = Math.floor(y1);
            if (pixelX >= 0 && pixelX < videoElement.videoWidth && pixelY >= 0 && pixelY < videoElement.videoHeight) {
                const pixelData = hiddenCtx.getImageData(pixelX, pixelY, 1, 1).data;
                hexColor = rgbToHex(pixelData[0], pixelData[1], pixelData[2]);
            }

            // 3. EKRANDAKİ UI PANELİNİ GÜNCELLE
            document.getElementById('live-wrist').innerText = liveWristCm;
            document.getElementById('live-color-box').style.backgroundColor = hexColor;

            // 4. HİZALAMA VE GERİ SAYIM KONTROLÜ
            // Bileğin (wrist.x ve wrist.y) ekranın ortasındaki kılavuz kutuya girip girmediğini kontrol et
            // Koordinatlar 0.0 ile 1.0 arasındadır. Merkez 0.5'tir.
            const isHandInBox = (wrist.x > 0.40 && wrist.x < 0.60 && wrist.y > 0.50 && wrist.y < 0.85);

            if (isHandInBox && !isCountingDown) {
                startCountdown();
            }

            // Eğer geri sayım başladıysa, verileri tampona (buffer) kaydet ki ortalamasını alalım
            if (isCountingDown) {
                measurementBuffer.push({
                    pixelDist: pixelDistance,
                    color: hexColor
                });
            }
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
}

// --- YENİ: GERİ SAYIM VE OTOMATİK GEÇİŞ FONKSİYONLARI ---
function startCountdown() {
    isCountingDown = true;
    measurementBuffer = []; // Önceki verileri temizle
    
    const overlay = document.getElementById('countdown-overlay');
    overlay.style.display = 'flex';
    let count = 3;
    overlay.innerText = count;

    const interval = setInterval(() => {
        count--;
        if (count > 0) {
            overlay.innerText = count;
        } else {
            clearInterval(interval);
            overlay.style.display = 'none';
            finalizeMeasurement();
        }
    }, 1000); // Her 1 saniyede (1000ms) bir çalışır
}

function finalizeMeasurement() {
    // 3 saniye boyunca toplanan piksel mesafelerinin ortalamasını alarak titreme hatasını yok et
    let totalPixelDist = 0;
    measurementBuffer.forEach(data => {
        totalPixelDist += data.pixelDist;
    });
    
    const averagePixelDist = totalPixelDist / measurementBuffer.length;
    
    // Verileri global objemize kaydet
    window.userData.wristPixelDistance = averagePixelDist;
    
    // Son okunan rengi al
    if (measurementBuffer.length > 0) {
        window.userData.skinColorHex = measurementBuffer[measurementBuffer.length - 1].color;
    }

    console.log("Ölçüm Kilitlendi! Ortalama Piksel:", averagePixelDist);
    
    // Kamerayı kapat ve otomatik olarak Trendyol Linki isteme (Asistana Sor) aşamasına geç!
    nextStep(4); 
}

// --- EKRAN YÖNETİMİ (WIZARD) ---
window.nextStep = function(stepNumber) {
    // 2. Adıma geçerken verileri kaydet
    if (stepNumber === 3) {
        window.userData.gender = document.getElementById('gender').value;
        window.userData.height = parseFloat(document.getElementById('height').value) || 175;
        window.userData.weight = parseFloat(document.getElementById('weight').value) || 70;
        startCamera(); // 3. adıma geçildiğinde kamerayı aç
    } else {
        stopCamera(); // Diğer adımlarda kamerayı kapat (Performans için)
    }

    // 4. Adıma (Sonuç) geçerken verileri topla ve API'ye at
    if (stepNumber === 4) {
        window.userData.productLink = document.getElementById('productLink').value;
        const resultBox = document.getElementById('result-box');
        
        resultBox.innerHTML = "<p>Yapay zeka saat linkini inceliyor ve bilek profilinize göre ölçümleri analiz ediyor... Lütfen bekleyin.</p>";
        
        // Backend'e istek at (fetch)
        fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(window.userData)
        })
        .then(response => response.json())
        .then(data => {
            console.log("Backend'den gelen cevap:", data);
            if(data.status === "success") {
                resultBox.innerHTML = `
                    <h3 style="color: #FF6B00; margin-top:0;">Analiz Tamamlandı</h3>
                    <p><strong>Öneri:</strong> ${data.recommendation}</p>
                    <p style="font-size: 0.85em; color: #888;">Çekilen Ürün Verisi: ${data.scraped_data.kasa_capi} / ${data.scraped_data.materyal}</p>
                `;
            } else {
                resultBox.innerHTML = `<p style="color: red;">Bir hata oluştu: ${data.detail}</p>`;
            }
        })
        .catch(error => {
            console.error('Hata:', error);
            resultBox.innerHTML = `<p style="color: red;">Sunucuya bağlanılamadı. Lütfen backend'in çalıştığından emin olun.</p>`;
        });
    }

    // UI Geçişi
    const allSteps = document.querySelectorAll('.wizard-step');
    allSteps.forEach(step => step.classList.remove('active'));
    
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) targetStep.classList.add('active');
};