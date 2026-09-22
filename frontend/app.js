// Dynamic Backend API URL (defaults to live Render backend, or local if running on localhost)
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const defaultApi = isLocal ? 'http://127.0.0.1:5000' : 'https://old-document-digitizer.onrender.com';
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('api')) {
    localStorage.setItem('backend_url', urlParams.get('api').replace(/\/$/, ''));
}
let BACKEND_URL = localStorage.getItem('backend_url') || defaultApi;

let selectedMethod = null; // 'upload', 'laptop', or 'mobile'
let currentProcessingMode = 'ai'; // 'ai' or 'fast'

function setProcessingMode(mode) {
    currentProcessingMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById('mode-btn-' + mode);
    if (activeBtn) activeBtn.classList.add('active');
}



// --- Navigation ---
function navigate(pageId) {
    // Update header active links
    document.querySelectorAll('.nav-home').forEach(el => el.classList.toggle('active', pageId === '1'));
    document.querySelectorAll('.nav-about').forEach(el => el.classList.toggle('active', pageId === 'about'));

    // Stop camera if we are navigating away from page 4a
    if (pageId !== '4a' && typeof stopLaptopCamera === 'function') {
        stopLaptopCamera();
    }
    // Hide all pages with a fade out effect
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.style.opacity = '0';
        setTimeout(() => {
            page.style.display = 'none';
            page.classList.remove('active');
        }, 300); // Wait for fade out
    });

    // Show target page with a fade in effect
    setTimeout(() => {
        const targetPage = document.getElementById('page-' + pageId);
        if (!targetPage) return;
        targetPage.style.display = 'block';
        
        // Trigger reflow for transition
        void targetPage.offsetWidth; 
        
        targetPage.style.opacity = '1';
        targetPage.classList.add('active');
        
        // Start camera if we are navigating to 4a
        if (pageId === '4a') {
            startLaptopCamera();
        }
    }, 300);
}

// --- Page 2 Logic ---
function selectMethod(method) {
    selectedMethod = method;
    
    // Enable the select button
    const btn = document.getElementById('btn-select-method');
    btn.disabled = false;
    
    // UI Feedback for selection
    const cards = document.querySelectorAll('.choice-card');
    cards.forEach(card => card.classList.remove('selected'));
    
    if (method === 'upload') {
        cards[0].classList.add('selected');
    } else {
        cards[1].classList.add('selected');
    }
}

function confirmMethod() {
    if (!selectedMethod) return;
    
    if (selectedMethod === 'upload') {
        navigate('3a');
    } else if (selectedMethod === 'camera') {
        navigate('3b');
    }
}

// --- Page 3B Logic ---
function selectCameraType(cameraType) {
    console.log("Selected camera type:", cameraType);
    selectedMethod = cameraType;
    if (cameraType === 'laptop') {
        navigate('4a');
    } else if (cameraType === 'mobile') {
        navigate('4b');
    }
}

// --- Page 3A Logic ---
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewImage = document.getElementById('image-preview');
            const previewText = document.getElementById('preview-text');
            const extractBtn = document.getElementById('btn-extract');
            
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            previewText.style.display = 'none';
            
            extractBtn.disabled = false;
        }
        reader.readAsDataURL(file);
    }
}

let extractionController = null;

function startExtraction(method) {
    selectedMethod = method;
    navigate('5');
    startLoadingAnimation();
    
    // Create new abort controller for this request
    extractionController = new AbortController();
    const signal = extractionController.signal;
    
    if (method === 'upload') {
        const fileInput = document.getElementById('file-upload');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            alert("No file selected.");
            navigate('3a');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('mode', currentProcessingMode);
        
        fetch(`${BACKEND_URL}/upload`, {
            method: 'POST',
            body: formData,
            signal: signal
        })
        .then(response => response.json())
        .then(data => displayResults(data))
        .catch(err => {
            if (err.name === 'AbortError') return;
            console.error(err); alert("Extraction failed."); navigate('3a'); 
        });
        
    } else if (method === 'laptop') {
        const canvas = document.getElementById('laptop-camera-canvas');
        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'laptop_capture.jpg');
            formData.append('mode', currentProcessingMode);
            
            fetch(`${BACKEND_URL}/upload`, {
                method: 'POST',
                body: formData,
                signal: signal
            })
            .then(response => response.json())
            .then(data => displayResults(data))
            .catch(err => {
                if (err.name === 'AbortError') return;
                console.error(err); alert("Extraction failed."); navigate('4a'); 
            });
        }, 'image/jpeg');
        
    } else if (method === 'mobile') {
        fetch(`${BACKEND_URL}/ipcamera/process?mode=${encodeURIComponent(currentProcessingMode)}`, {
            method: 'POST',
            signal: signal
        })
        .then(response => response.json())
        .then(data => displayResults(data))
        .catch(err => {
            if (err.name === 'AbortError') return;
            console.error(err); alert("Extraction failed."); navigate('4c'); 
        });
    }
}

function cancelExtraction() {
    if (extractionController) {
        extractionController.abort();
    }
    clearInterval(loadingInterval);
    
    // Determine where to go back to based on the method
    if (selectedMethod === 'upload') {
        navigate('3a');
    } else if (selectedMethod === 'laptop') {
        navigate('4a');
    } else {
        navigate('4c');
    }
}

// --- Page 5 Logic (Loading) ---
let loadingInterval;
function startLoadingAnimation() {
    const isFast = (currentProcessingMode === 'fast');
    const statuses = isFast ? [
        "Analyzing Image Layout...",
        "Running High-Speed PaddleOCR...",
        "Generating Instant Transcription..."
    ] : [
        "Segmenting Document Strokes...",
        "Running Neural Vision OCR...",
        "Deciphering Cursive Sequences...",
        "Refining Grammar & Spelling with AI...",
        "Generating Formatted Document..."
    ];
    
    let step = 0;
    const statusText = document.getElementById('loading-status-text');
    const progressBar = document.getElementById('loading-progress');
    const nodes = document.querySelectorAll('.progress-nodes .node');
    
    clearInterval(loadingInterval);
    
    // Reset
    progressBar.style.width = '0%';
    nodes.forEach(n => n.classList.remove('active'));
    nodes[0].classList.add('active');
    statusText.innerText = statuses[0];
    
    const intervalMs = isFast ? 500 : 1800;
    const progressPerStep = Math.floor(90 / statuses.length);

    loadingInterval = setInterval(() => {
        step++;
        if (step < statuses.length) {
            statusText.innerText = statuses[step];
            progressBar.style.width = Math.min(90, step * progressPerStep) + '%';
            nodes.forEach((n, idx) => {
                if (idx <= step) n.classList.add('active');
            });
        }
    }, intervalMs);
}

// --- Page 6 Logic (Results) ---
let currentResult = null;

function displayResults(data) {
    clearInterval(loadingInterval);
    
    // Force 100% progress before switching
    document.getElementById('loading-progress').style.width = '100%';
    document.querySelectorAll('.progress-nodes .node').forEach(n => n.classList.add('active'));
    document.getElementById('loading-status-text').innerText = "Complete!";
    
    if (data.success === false) {
        alert("Error: " + data.message);
        navigate('1'); // back to home
        return;
    }
    
    currentResult = data;
    
    setTimeout(() => {
        const rawEl = document.getElementById('raw-ocr-text');
        const corrEl = document.getElementById('corrected-ocr-text');
        if (rawEl) rawEl.innerText = data.raw_text || "No text detected.";
        if (corrEl) corrEl.innerText = data.corrected_text || data.raw_text || "No text generated.";

        navigate('6');
    }, 600);
}

function downloadResult(format) {
    if (!currentResult) return;
    const content = format === 'txt' ? 
        "--- Original ---\n" + currentResult.raw_text + "\n\n--- Corrected ---\n" + currentResult.corrected_text :
        JSON.stringify(currentResult, null, 2);
        
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "extraction_result." + format;
    a.click();
    URL.revokeObjectURL(url);
}

function copyResult() {
    if (!currentResult) return;
    const text = currentResult.corrected_text || currentResult.raw_text;
    navigator.clipboard.writeText(text).then(() => {
        alert("Copied to clipboard!");
    });
}

function resetApp() {
    currentResult = null;
    navigate('2');
}

// --- Page 4A Logic (Laptop Camera) ---
let laptopStream = null;

async function startLaptopCamera() {
    try {
        const video = document.getElementById('laptop-camera-feed');
        // Ensure video is playing and canvas is hidden
        video.style.display = 'block';
        document.getElementById('laptop-camera-canvas').style.display = 'none';
        
        laptopStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = laptopStream;
        
        // Reset buttons
        document.getElementById('btn-extract-laptop').disabled = true;
        document.querySelector('#btn-capture-laptop .button-text').innerText = 'Capture';
    } catch (err) {
        console.error("Error accessing laptop camera:", err);
        alert("Could not access camera. Please check permissions.");
    }
}

function stopLaptopCamera() {
    if (laptopStream) {
        laptopStream.getTracks().forEach(track => track.stop());
        laptopStream = null;
    }
}

function captureLaptopImage() {
    const video = document.getElementById('laptop-camera-feed');
    const canvas = document.getElementById('laptop-camera-canvas');
    const extractBtn = document.getElementById('btn-extract-laptop');
    const captureText = document.querySelector('#btn-capture-laptop .button-text');
    
    if (captureText.innerText === 'Capture') {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Hide video, show canvas
        video.style.display = 'none';
        canvas.style.display = 'block';
        
        extractBtn.disabled = false;
        captureText.innerText = 'Retake';
    } else {
        // Retake mode
        video.style.display = 'block';
        canvas.style.display = 'none';
        extractBtn.disabled = true;
        captureText.innerText = 'Capture';
    }
}

// --- Page 4B Logic (Mobile IP Setup) ---
function connectMobileCamera() {
    let rawIp = document.getElementById('ip-camera-url').value.trim();
    if (!rawIp) {
        alert("Please enter the IP Webcam URL.");
        return;
    }
    
    // The backend wants just the IP and Port (e.g., 192.168.1.100:8080)
    // Send it to the backend to connect
    const connectBtn = document.querySelector('.connect-btn .button-text') || document.querySelector('.extract-btn .button-text');
    if (connectBtn) connectBtn.innerText = "Connecting...";
    
    fetch(`${BACKEND_URL}/ipcamera/connect_phone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip: rawIp })
    })
    .then(res => res.json())
    .then(data => {
        if (connectBtn) connectBtn.innerText = "Connect Camera";
        if (data.success) {
            document.getElementById('mobile-video-preview').src = `${BACKEND_URL}/ipcamera/video_feed?` + new Date().getTime();
            navigate('4c');
        } else {
            alert("Failed to connect: " + data.message);
        }
    })
    .catch(err => {
        if (connectBtn) connectBtn.innerText = "Connect Camera";
        console.error(err);
        alert("Could not connect to backend.");
    });
}

function disconnectMobileCamera() {
    fetch(`${BACKEND_URL}/ipcamera/disconnect_phone`)
    .then(() => {
        document.getElementById('mobile-video-preview').src = "";
        navigate('3b');
    });
}

// --- Dynamic API Server Configuration & Health Check ---
function promptBackendUrl() {
    const current = localStorage.getItem('backend_url') || BACKEND_URL;
    const newUrl = prompt("Enter your Backend API URL (e.g. Hugging Face Space URL):\nExample: https://yourusername-old-document-digitizer.hf.space", current);
    if (newUrl !== null && newUrl.trim() !== '') {
        const cleaned = newUrl.trim().replace(/\/$/, '');
        localStorage.setItem('backend_url', cleaned);
        BACKEND_URL = cleaned;
        checkBackendHealth();
        alert("Backend URL updated to:\n" + cleaned);
    }
}

async function checkBackendHealth() {
    const dot = document.getElementById('apiStatusDot');
    const label = document.getElementById('apiStatusLabel');
    if (!dot) return;
    try {
        const res = await fetch(`${BACKEND_URL}/`, { method: 'GET', signal: AbortSignal.timeout(4000) });
        if (res.ok) {
            dot.style.background = '#22c55e';
            if (label) label.textContent = 'API Connected';
        } else {
            dot.style.background = '#f59e0b';
            if (label) label.textContent = 'API Error';
        }
    } catch (e) {
        dot.style.background = '#ef4444';
        if (label) label.textContent = 'API Offline';
    }
}

// Initial health check
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkBackendHealth);
} else {
    checkBackendHealth();
}


