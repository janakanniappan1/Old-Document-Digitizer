import os
import gc

# ── Cloud Free Tier Low-Memory & Single-Thread Enforcement ────────────────────
# Must be set before cv2, paddle, or numpy initialize C++/MKL/OpenMP threads
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", "0.05")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from flask import Flask, jsonify
from flask_cors import CORS

from routes.upload import upload_bp
from routes.camera import camera_bp
from routes.ipcamera import ipcamera_bp
from routes.result import result_bp

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "temp")

# Limit uploads to 50 MB (protects against oversized file attacks)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# ── CORS ───────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS env var in production to your frontend domain(s).
#   e.g.  ALLOWED_ORIGINS=https://old-document-digitizer.vercel.app
# If unset, defaults to "*" (allow all) for development convenience.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
if _raw_origins:
    _allowed = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    _allowed = "*"

CORS(app, origins=_allowed, methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Type"],
     supports_credentials=False)

# ── Blueprints ────────────────────────────────────────────────────────────────
app.register_blueprint(upload_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(ipcamera_bp, url_prefix="/ipcamera")
app.register_blueprint(result_bp)


# ── Global error handlers ─────────────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"success": False, "error": "File too large. Maximum allowed size is 50 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


@app.teardown_request
def cleanup_memory(exception=None):
    # Free temporary memory chunks from request processing
    gc.collect()


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    llm_name = "Ollama (Qwen2.5)"
    if os.environ.get("GROQ_API_KEY"):
        llm_name = f"Groq Cloud AI ({os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')})"
    elif os.environ.get("GEMINI_API_KEY"):
        llm_name = "Google Gemini Cloud AI"

    return jsonify({
        "project": "Old Document Digitizer",
        "backend": "Running",
        "version": "v2.3-lowmem",
        "OCR": "PaddleOCR (Low-Memory Mode)",
        "LLM": llm_name,
        "status": "Ready"
    })


# ── Dev-only test pages (not served in production) ────────────────────────────
# NOTE: These routes serve raw HTML files located at the project root.
# They are useful during local development only. In production, serve
# the frontend/ directory via Nginx and remove or gate these routes.
if os.environ.get("FLASK_ENV") != "production":
    PARENT_DIR = os.path.dirname(BASE_DIR)

    @app.route("/test")
    def test_page():
        path = os.path.join(PARENT_DIR, "test_camera.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html"}
        except FileNotFoundError:
            return jsonify({"error": "Test page not found"}), 404

    @app.route("/test_ip")
    def test_ip_page():
        path = os.path.join(PARENT_DIR, "test_ipcamera.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html"}
        except FileNotFoundError:
            return jsonify({"error": "Test page not found"}), 404

    @app.route("/test_upload")
    def test_upload_page():
        path = os.path.join(PARENT_DIR, "test_upload.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html"}
        except FileNotFoundError:
            return jsonify({"error": "Test page not found"}), 404


# ── Entry point (development only — use Gunicorn in production) ───────────────
if __name__ == "__main__":
    # debug=False is intentional. Never enable debug=True in production.
    # In production, run:
    #   gunicorn -w 1 -b 0.0.0.0:5000 --timeout 300 app:app
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    app.run(debug=False, port=port, host=host, threaded=True)