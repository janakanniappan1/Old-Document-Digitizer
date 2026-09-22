import os
import json
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ==========================================
# CREATE FOLDER IF NOT EXISTS
# ==========================================
def create_folder(path):
    os.makedirs(path, exist_ok=True)


# ==========================================
# GENERATE UNIQUE BASE NAME / FILE NAME
# ==========================================
def generate_base_name():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    return f"{timestamp}_{unique}"


def generate_filename(extension):
    return f"{generate_base_name()}.{extension}"


# ==========================================
# GET CURRENT DATE & TIME
# ==========================================
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================
# SAVE TEXT FILE
# ==========================================
def save_text(path, text):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        logger.error("Failed to save text file %s: %s", path, e)


# ==========================================
# SAVE JSON FILE
# ==========================================
def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logger.error("Failed to save JSON file %s: %s", path, e)


# ==========================================
# ALLOWED IMAGE EXTENSIONS
# ==========================================
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


# ==========================================
# FORMAT OCR RESPONSE
# NOTE: All paths stored are basenames only — never absolute server paths.
# ==========================================
def response_json(raw_text, corrected_text, image_path, txt_path, json_path, crnn_text=""):
    return {
        "success": True,
        "status": "success",
        "raw_text": raw_text,
        "crnn_text": crnn_text,
        "corrected_text": corrected_text,
        # Store only the filename, not the full server path (security: avoid path disclosure)
        "image_file": os.path.basename(image_path),
        "text_file": os.path.basename(txt_path),
        "json_file": os.path.basename(json_path),
        "created_at": get_timestamp()
    }