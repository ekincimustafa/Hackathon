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
        console.error("Kamera açılamadı:", error);
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
            
            // Çizimleri kalınlaştırdık (İnce ve cılız görünme sorunu çözüldü)
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2, radius: 5 });

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
    
    if (method === 'camera') {
        window.userData.gender = document.getElementById('gender').value;
        window.userData.height = parseFloat(document.getElementById('height').value) || 175;
        window.userData.weight = parseFloat(document.getElementById('weight').value) || 70;
        nextStep(3);
    } else {
        const manualVal = parseFloat(document.getElementById('manual-wrist').value);
        if(!manualVal) {
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

window.nextStep = function(stepNumber) {
    if (stepNumber === 3) {
        startCamera(); 
    } else {
        stopCamera(); 
    }

    if (stepNumber === 5) {
        window.userData.productLink = document.getElementById('productLink').value;
        if (!window.userData.productLink) {
            alert("Lütfen analiz için bir saat linki yapıştırın.");
            return; 
        }
        console.log("Yapay Zekaya Giden Veri:", window.userData);
        // İleride buraya FastAPI fetch isteğini ekleyeceğiz
    }

    const allSteps = document.querySelectorAll('.wizard-step');
    allSteps.forEach(step => step.classList.remove('active'));
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) targetStep.classList.add('active');
};