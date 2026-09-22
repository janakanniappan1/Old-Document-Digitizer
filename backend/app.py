import os
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
# For local development: allow all origins.
# For production: set ALLOWED_ORIGINS env var to your domain(s).
#   e.g.  ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
if _raw_origins:
    _allowed = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    CORS(app, origins=_allowed)
else:
    # Development fallback — allow everything locally
    CORS(app)

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


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "project": "Old Document Digitizer",
        "backend": "Running",
        "OCR": "PaddleOCR",
        "LLM": "Qwen2.5",
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
    app.run(debug=False, port=5000, host="127.0.0.1", threaded=True)