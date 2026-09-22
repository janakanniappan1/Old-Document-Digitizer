import os
import json
import logging
from werkzeug.utils import secure_filename

from flask import Blueprint, jsonify, send_from_directory

logger = logging.getLogger(__name__)

result_bp = Blueprint("result", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

# In-memory latest result.
# NOTE: This is intentionally a single in-memory value.
# It is reset on server restart. For persistent, multi-user history
# a database (e.g. PostgreSQL) would be required. The filesystem
# JSON files in /outputs serve as the persistent record.
latest_result = {}


# ==========================================
# Save Latest Result (called by processor)
# ==========================================
def update_result(data):
    global latest_result
    latest_result = data


# ==========================================
# Latest Result
# ==========================================
@result_bp.route("/latest_result")
def latest_result_api():
    if not latest_result:
        return jsonify({"success": False, "error": "No result available yet."}), 404
    return jsonify(latest_result)


# ==========================================
# OCR History (read from filesystem JSON files)
# ==========================================
@result_bp.route("/history")
def history():
    history_data = []

    if not os.path.exists(OUTPUT_FOLDER):
        return jsonify(history_data)

    try:
        files = sorted(os.listdir(OUTPUT_FOLDER), reverse=True)
    except OSError as e:
        logger.error("Could not list output folder: %s", e)
        return jsonify(history_data)

    for filename in files:
        if not filename.endswith(".json"):
            continue

        # Guard: only allow safe filenames
        safe_name = secure_filename(filename)
        if safe_name != filename:
            continue

        path = os.path.join(OUTPUT_FOLDER, safe_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                history_data.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping malformed history file %s: %s", filename, e)
            continue

    return jsonify(history_data)


# ==========================================
# Download TXT
# ==========================================
@result_bp.route("/download/txt/<filename>")
def download_txt(filename):
    return _safe_download(filename, expected_ext=".txt")


# ==========================================
# Download JSON
# ==========================================
@result_bp.route("/download/json/<filename>")
def download_json(filename):
    return _safe_download(filename, expected_ext=".json")


# ==========================================
# Download Image
# ==========================================
@result_bp.route("/download/image/<filename>")
def download_image(filename):
    return _safe_download(filename, expected_ext=None)  # Accept .jpg/.jpeg/.png


# ==========================================
# Safe Download Helper (path traversal guard)
# ==========================================
def _safe_download(filename, expected_ext=None):
    safe_name = secure_filename(filename)

    # Reject if filename changed after sanitisation (path traversal attempt)
    if safe_name != filename or not safe_name:
        return jsonify({"success": False, "error": "Invalid filename."}), 400

    # Enforce extension
    if expected_ext and not safe_name.lower().endswith(expected_ext):
        return jsonify({"success": False, "error": f"Invalid file type. Expected {expected_ext}."}), 400

    full_path = os.path.join(OUTPUT_FOLDER, safe_name)

    # Resolve to absolute path and verify it is inside OUTPUT_FOLDER
    real_output = os.path.realpath(OUTPUT_FOLDER)
    real_path = os.path.realpath(full_path)
    if not real_path.startswith(real_output + os.sep):
        return jsonify({"success": False, "error": "Access denied."}), 403

    if not os.path.isfile(real_path):
        return jsonify({"success": False, "error": "File not found."}), 404

    return send_from_directory(OUTPUT_FOLDER, safe_name, as_attachment=True)