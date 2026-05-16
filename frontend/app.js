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
let measurementBuffer = []; // 3 saniye boyunca toplanacak oran verileri
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
        numHands: 1
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

// --- YARDIMCI MATEMATİK FONKSİYONLARI ---
function rgbToHex(r, g, b) {
    return "#" + (1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1).toUpperCase();
}

function getDistance(p1, p2, width, height) {
    return Math.sqrt(Math.pow((p2.x - p1.x) * width, 2) + Math.pow((p2.y - p1.y) * height, 2));
}

// --- ZEKİ ARALIK HESAPLAMA (HEURISTIC ALGORITHM) ---
function calculateWristRange(handRatio, height, weight, gender) {
    // 1. Temel Başlangıç Değeri
    let baseWrist = 16.0; 
    if (gender === 'male') baseWrist = 17.0;
    if (gender === 'female') baseWrist = 15.0;

    // 2. Vücut Kitle İndeksi (VKİ) Etkisi
    const bmi = weight / Math.pow(height / 100, 2);
    let bmiModifier = 0;
    if (bmi > 25) bmiModifier = (bmi - 25) * 0.15;
    if (bmi < 18.5) bmiModifier = (bmi - 18.5) * 0.15; 
    
    // VKİ etkisini sınırla
    bmiModifier = Math.max(-2.0, Math.min(2.5, bmiModifier));

    // 3. Anatomik El Oranı Etkisi (Ortalama oran 0.70 civarıdır)
    let ratioModifier = (handRatio - 0.70) * 8; 

    // Tahmini net değeri bul
    let estimatedWrist = baseWrist + bmiModifier + ratioModifier;

    // Kullanıcıya sunulacak 1 cm'lik aralığı (Range) oluştur
    let lowerBound = (estimatedWrist - 0.5).toFixed(1);
    let upperBound = (estimatedWrist + 0.5).toFixed(1);
    
    return `${lowerBound} - ${upperBound}`;
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
            
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 1, radius: 3 });

            // Z ekseninden bağımsız anatomik oran hesaplama
            const wrist = landmarks[0];
            const middleMcp = landmarks[9]; // El uzunluğu referansı
            const indexMcp = landmarks[5]; // Avuç genişliği sol sınır
            const pinkyMcp = landmarks[17]; // Avuç genişliği sağ sınır

            const handLength = getDistance(wrist, middleMcp, videoElement.videoWidth, videoElement.videoHeight);
            const palmWidth = getDistance(indexMcp, pinkyMcp, videoElement.videoWidth, videoElement.videoHeight);
            
            const handRatio = palmWidth / handLength;

            // VKİ ve Oran kullanarak anlık tahmini aralığı bul
            const liveRange = calculateWristRange(handRatio, window.userData.height, window.userData.weight, window.userData.gender);

            // Ten Rengi
            let hexColor = "#000000";
            const pixelX = Math.floor(wrist.x * videoElement.videoWidth);
            const pixelY = Math.floor(wrist.y * videoElement.videoHeight);
            if (pixelX >= 0 && pixelX < videoElement.videoWidth && pixelY >= 0 && pixelY < videoElement.videoHeight) {
                const pixelData = hiddenCtx.getImageData(pixelX, pixelY, 1, 1).data;
                hexColor = rgbToHex(pixelData[0], pixelData[1], pixelData[2]);
            }

            // Ekrana canlı yazdır
            document.getElementById('live-wrist').innerText = liveRange;
            document.getElementById('live-color-box').style.backgroundColor = hexColor;

            // Hizalama Kontrolü
            const isHandInBox = (wrist.x > 0.40 && wrist.x < 0.60 && wrist.y > 0.40 && wrist.y < 0.85);

            if (isHandInBox && !isCountingDown) {
                startCountdown();
            } else if (!isHandInBox && isCountingDown) {
                cancelCountdown();
            }

            // Geri sayım sürüyorsa verileri tampona (buffer) at
            if (isCountingDown) {
                measurementBuffer.push({ ratio: handRatio, color: hexColor });
            }
        } else {
            if (isCountingDown) cancelCountdown();
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
}

// --- GERİ SAYIM YÖNETİMİ ---
function startCountdown() {
    isCountingDown = true;
    measurementBuffer = []; 
    
    const overlay = document.getElementById('countdown-overlay');
    overlay.style.display = 'flex';
    let count = 3;
    overlay.innerText = count;

    countdownInterval = setInterval(() => {
        count--;
        if (count > 0) {
            overlay.innerText = count;
        } else {
            clearInterval(countdownInterval);
            overlay.style.display = 'none';
            finalizeMeasurement();
        }
    }, 1000);
}

function cancelCountdown() {
    isCountingDown = false;
    clearInterval(countdownInterval);
    document.getElementById('countdown-overlay').style.display = 'none';
    console.log("El hizadan çıktı, ölçüm iptal edildi.");
}

function finalizeMeasurement() {
    let totalRatio = 0;
    measurementBuffer.forEach(data => totalRatio += data.ratio);
    const averageRatio = totalRatio / measurementBuffer.length;
    
    window.userData.wristRangeStr = calculateWristRange(averageRatio, window.userData.height, window.userData.weight, window.userData.gender);
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
        // İleride buraya fetch isteğini ekleyeceğiz
    }

    const allSteps = document.querySelectorAll('.wizard-step');
    allSteps.forEach(step => step.classList.remove('active'));
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) targetStep.classList.add('active');
};