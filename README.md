# 📜 Old Document Digitizer

A web-based AI OCR system for digitizing old handwritten documents, historical manuscripts, and cursive text using a local multi-engine AI pipeline.

---

## 🧠 AI Pipeline

```
Image Input
  ↓
PaddleOCR  — multi-scale text detection + recognition
  ↓
CRNN       — custom TensorFlow CTC neural network (verification pass)
  ↓
Qwen2.5:7b — Ollama LLM for linguistic error correction
  ↓
Structured Output (JSON + TXT + Annotated Image)
```

All inference runs **locally on your machine**. No images or text are ever sent to external servers.

---

## 🌐 Input Methods

| Method | Description |
|--------|-------------|
| **File Upload** | Upload JPG, PNG, BMP, TIFF, or WebP images |
| **Laptop Webcam** | Browser-side camera capture via `getUserMedia` |
| **IP Camera (Phone)** | MJPEG stream from a phone camera app on the same network |

> ⚠️ **Webcam note**: The laptop camera is captured entirely in the browser using the [MediaDevices API](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia). No webcam feed is sent to the server — only the captured frame. The server-side webcam routes (`/start`, `/video_feed`) are for a physical camera attached to the server machine.

---

## 🚀 Local Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 or 3.11 | 3.12+ not supported by PaddlePaddle |
| Ollama | Latest | [ollama.com](https://ollama.com) |
| Qwen2.5:7b | — | Downloaded via Ollama |

### 1. Clone and set up Python environment

```bash
git clone <repo-url>
cd Old-Document-Digitizer/backend

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install and configure Ollama

```bash
# Install Ollama: https://ollama.com/download

# Pull the required LLM model
ollama pull qwen2.5:7b

# Verify Ollama is running
ollama list
```

### 3. Configure environment variables (optional)

```bash
cp .env.example .env
# Edit .env as needed
```

### 4. Run the Flask backend

```bash
# Development (from backend/ directory):
python app.py

# Or with Gunicorn (recommended even locally):
gunicorn -w 1 -b 127.0.0.1:5000 --timeout 300 app:app
```

### 5. Open the frontend

Open `frontend/index.html` directly in your browser, **or** serve it with any static server:

```bash
# With Python's built-in server (from frontend/ directory):
python -m http.server 8080
# Then open: http://localhost:8080
```

---

## 🏭 Production Architecture

```
Browser (HTTPS)
     ↓
  Nginx  ──────────────── serves frontend/ static files
     ↓
  Gunicorn (port 5000, -w 1)
     ↓
  Flask  ─── PaddleOCR
          ├── CRNN (TensorFlow)
          └── Ollama (localhost:11434) → Qwen2.5:7b
```

### Production Nginx config snippet

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    root /opt/old-doc-digitizer/frontend;
    index index.html;

    location / {
        try_files $uri /index.html;
    }

    # Proxy API calls to Flask
    location ~ ^/(upload|start|stop|capture|process|video_feed|camera_status|ipcamera|latest_result|history|download) {
        proxy_pass http://127.0.0.1:5000;
        proxy_read_timeout 300s;
        client_max_body_size 50M;
    }
}
```

### Run as a systemd service

```ini
# /etc/systemd/system/old-doc-digitizer.service
[Unit]
Description=Old Document Digitizer
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/old-doc-digitizer/backend
EnvironmentFile=/opt/old-doc-digitizer/.env
ExecStart=/opt/old-doc-digitizer/backend/venv/bin/gunicorn \
    -w 1 -b 127.0.0.1:5000 --timeout 300 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## ⚙️ Environment Variables

See [`.env.example`](.env.example) for all available settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model to use for text correction |
| `ALLOWED_ORIGINS` | *(empty = allow all)* | Comma-separated CORS origins for production |
| `FLASK_ENV` | `development` | Set to `production` to disable dev test routes |

---

## 📁 Project Structure

```
Old-Document-Digitizer/
├── frontend/           # Static UI (HTML + JS + CSS — no framework)
├── backend/
│   ├── app.py          # Flask entry point
│   ├── requirements.txt
│   ├── routes/         # API blueprints
│   ├── services/       # OCR / LLM / CRNN / image processing
│   ├── utils/          # Helpers
│   ├── model/          # CRNN model files (crnn_ctc.keras, vocab.json)
│   ├── outputs/        # Auto-generated OCR results (gitignored)
│   └── temp/           # Temporary uploads (gitignored)
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚠️ Important Limitations

### Single worker required
Due to in-memory model instances (PaddleOCR, TensorFlow, Ollama client) and camera state globals, always run with **exactly 1 Gunicorn worker** (`-w 1`). Multiple workers will create redundant model copies, exhaust RAM, and corrupt camera state.

### Webcam on cloud servers
`cv2.VideoCapture(0)` requires a **physical camera attached to the server**. Cloud VPS machines do not have webcams. On cloud deployments:
- **File upload**: works everywhere ✅
- **Browser webcam** (laptop camera tab): captures in browser via JavaScript, works everywhere ✅
- **IP camera**: works when phone and server are on the same network ✅
- **Server-side `/start` webcam**: only works on a machine with a physical camera ⚠️

### No multi-user database
The current architecture stores results on the filesystem. The "latest result" is an in-memory Python dict (lost on restart). For multi-user production deployment, add a database (e.g., PostgreSQL) and persistent storage (e.g., S3 or a volume).

### RAM requirements
Qwen2.5:7b requires ~8–10 GB RAM. Minimum server RAM: **16 GB**.

---

## 📦 Hardware Requirements (Production)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50+ GB |
| GPU | Not required | Optional (speeds up PaddleOCR) |

---

## 📄 License

This project was built by **The AlphaBrothers** team.
