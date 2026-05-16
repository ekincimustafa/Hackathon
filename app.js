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
        alert("Modeller henüz yüklenmedi, lütfen 1-2 saniye bekleyin.");
        return;
    }

    // Cihazın mobil olup olmadığını basitçe tespit et
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Mobilde arka kamera, masaüstünde ön kamera
    const facingMode = isMobile ? "environment" : "user";
    
    // Masaüstünde ayna efekti ver (ön kamera kullanıldığı için)
    if (!isMobile) {
        videoElement.style.transform = "scaleX(-1)";
        canvasElement.style.transform = "scaleX(-1)";
    } else {
        videoElement.style.transform = "none";
        canvasElement.style.transform = "none";
    }

    const constraints = {
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: facingMode },
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
            
            // İskeleti çiz (Frictionless UX için yeşil çizgiler)
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#FF6B00", lineWidth: 2 });
            drawingUtils.drawLandmarks(landmarks, { color: "#FFFFFF", lineWidth: 1, radius: 3 });

            // 0: Bilek, 5: İşaret parmağı kökü
            const wrist = landmarks[0];
            const indexMcp = landmarks[5];

            // Pikselleri hesapla
            const x1 = wrist.x * videoElement.videoWidth;
            const y1 = wrist.y * videoElement.videoHeight;
            const x2 = indexMcp.x * videoElement.videoWidth;
            const y2 = indexMcp.y * videoElement.videoHeight;

            window.userData.wristPixelDistance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));

            // Ten rengini al
            const pixelX = Math.floor(x1);
            const pixelY = Math.floor(y1);
            if (pixelX >= 0 && pixelX < videoElement.videoWidth && pixelY >= 0 && pixelY < videoElement.videoHeight) {
                const pixelData = hiddenCtx.getImageData(pixelX, pixelY, 1, 1).data;
                window.userData.skinColorHex = rgbToHex(pixelData[0], pixelData[1], pixelData[2]);
            }
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
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
        console.log("Yapay Zekaya Gönderilecek Veriler:", window.userData);
        document.getElementById('result-box').innerHTML = "<p>Yapay zeka saat linkini inceliyor ve bilek profilinize göre ölçümleri analiz ediyor... Lütfen bekleyin.</p>";
        // TODO: FastAPI Backend'e fetch isteği burada atılacak
    }

    // UI Geçişi
    const allSteps = document.querySelectorAll('.wizard-step');
    allSteps.forEach(step => step.classList.remove('active'));
    
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) targetStep.classList.add('active');
};