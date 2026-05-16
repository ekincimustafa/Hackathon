// Sayfalar arası geçiş fonksiyonu
function nextStep(stepNumber) {
    // Tüm adımların 'active' sınıfını kaldır
    const allSteps = document.querySelectorAll('.wizard-step');
    allSteps.forEach(step => {
        step.classList.remove('active');
    });

    // İstenen adımı göster
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) {
        targetStep.classList.add('active');
    }

    // Eğer 3. adıma (Kamera ekranına) geçildiyse MediaPipe'ı tetikleme mantığı buraya gelecek
    if (stepNumber === 3) {
        console.log("Kamera analizi başlatılıyor...");
        // TODO: startCamera() fonksiyonu çağrılacak
    }
    
    // Eğer 4. adıma (Sonuç ekranına) geçildiyse FastAPI'ye istek atma mantığı buraya gelecek
    if (stepNumber === 4) {
        console.log("Backend'e veri gönderiliyor, yapay zeka düşünmeye başladı...");
        // TODO: fetch('http://127.0.0.1:8000/analyze', {...})
    }
}