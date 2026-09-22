import logging

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from services.processor import process_image
from utils.helper import allowed_file

logger = logging.getLogger(__name__)

upload_bp = Blueprint("upload", __name__)

# Maximum image dimension guard (protects against excessively large images)
MAX_IMAGE_DIM = 8000


@upload_bp.route("/upload", methods=["POST"])
def upload():
    # ── 1. Presence check ─────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    # ── 2. Extension check ────────────────────────────────────────────────────
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Allowed: jpg, jpeg, png, bmp, tiff, webp."
        }), 400

    # ── 3. Decode image from memory (no temp file needed) ─────────────────────
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
    except Exception:
        return jsonify({"success": False, "error": "Failed to read uploaded file."}), 400

    if file_bytes.size == 0:
        return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({
            "success": False,
            "error": "Could not decode image. The file may be corrupted or in an unsupported format."
        }), 400

    # ── 4. Sanity check image dimensions ──────────────────────────────────────
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return jsonify({"success": False, "error": "Decoded image has zero dimensions."}), 400

    if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
        return jsonify({
            "success": False,
            "error": f"Image dimensions ({w}x{h}) exceed maximum allowed ({MAX_IMAGE_DIM}px per side)."
        }), 400

    # ── 5. Processing mode ────────────────────────────────────────────────────
    mode = (request.form.get("mode") or request.args.get("mode", "ai")).strip().lower()
    use_llm = (mode != "fast")

    # ── 6. Run OCR pipeline ───────────────────────────────────────────────────
    try:
        result = process_image(image, use_llm=use_llm)
        return jsonify(result)
    except Exception as e:
        logger.exception("OCR pipeline error during /upload")
        # Return a safe error message — do NOT expose the raw exception string
        return jsonify({
            "success": False,
            "error": "Processing failed. Please check server logs for details."
        }), 500