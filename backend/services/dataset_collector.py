import os
import csv
import cv2
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_DIR = os.path.join(BASE_DIR, "model", "feedback_data")
CROPS_DIR = os.path.join(FEEDBACK_DIR, "crops")
LABELS_CSV = os.path.join(FEEDBACK_DIR, "labels.csv")


def init_feedback_storage():
    os.makedirs(CROPS_DIR, exist_ok=True)
    if not os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "confidence", "timestamp"])


def harvest_lines(image, lines, label_text=""):
    """
    Harvests cropped text line images and logs them with their labels into feedback_data.
    Skipped in production cloud hosting to conserve memory and disk inodes.
    """
    if os.environ.get("FLASK_ENV") == "production" or os.environ.get("ENABLE_DATASET_HARVEST", "false").lower() != "true":
        return 0

    if image is None or not lines:
        return 0

    init_feedback_storage()
    h, w = image.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_count = 0

    records = []

    for idx, line in enumerate(lines):
        text = line.get("text", "").strip()
        bbox = line.get("bbox", [])
        conf = float(line.get("confidence", 0.0))

        if not text or len(bbox) < 4:
            continue

        # Extract bounding box
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]

        # Add 3px margin
        x_min = max(0, min(xs) - 3)
        x_max = min(w, max(xs) + 3)
        y_min = max(0, min(ys) - 3)
        y_max = min(h, max(ys) + 3)

        if (x_max - x_min) < 10 or (y_max - y_min) < 8:
            continue

        crop = image[y_min:y_max, x_min:x_max]
        crop_filename = f"crop_{timestamp}_{idx:03d}.png"
        crop_path = os.path.join(CROPS_DIR, crop_filename)

        cv2.imwrite(crop_path, crop)
        records.append([crop_filename, text, f"{conf:.2f}", timestamp])
        saved_count += 1

    if records:
        with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(records)

    return saved_count
