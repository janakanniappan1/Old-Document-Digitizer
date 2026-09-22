import os
import logging

import cv2

from services.instances import paddle_ocr, llm, crnn
from services.dataset_collector import harvest_lines
from routes.result import update_result, OUTPUT_FOLDER
from utils.helper import create_folder, generate_base_name, save_text, save_json, response_json

logger = logging.getLogger(__name__)


def process_image(image, use_llm=True):
    # Ensure output directory exists
    create_folder(OUTPUT_FOLDER)

    # ── Run OCR ───────────────────────────────────────────────────────────────
    result = paddle_ocr.extract_text(image)
    raw_text = result["text"]
    lines = result["lines"]

    # ── Draw bounding boxes on annotated output image ─────────────────────────
    output_img = paddle_ocr.draw_boxes(image, lines)

    # ── Generate unified, unique base filename ────────────────────────────────
    base_name = generate_base_name()
    img_name = f"{base_name}.jpg"
    txt_name = f"{base_name}.txt"
    json_name = f"{base_name}.json"

    img_path = os.path.join(OUTPUT_FOLDER, img_name)
    txt_path = os.path.join(OUTPUT_FOLDER, txt_name)
    json_path = os.path.join(OUTPUT_FOLDER, json_name)

    # ── Save annotated image ─────────────────────────────────────────────────
    cv2.imwrite(img_path, output_img)

    # ── CRNN verification pass ────────────────────────────────────────────────
    # Currently crnn.correct_text() passes text through unchanged.
    # When a better CRNN model is available this step will perform
    # actual word-level correction.
    crnn_text = crnn.correct_text(raw_text)

    # ── LLM correction (skipped in fast mode) ─────────────────────────────────
    if use_llm:
        corrected_text = llm.correct_text(crnn_text)
    else:
        corrected_text = crnn_text

    # ── Save text output ──────────────────────────────────────────────────────
    text_content = f"--- Original OCR ---\n{raw_text}\n\n--- AI Corrected ---\n{corrected_text}"
    save_text(txt_path, text_content)

    # ── Harvest feedback crops for future CRNN retraining ─────────────────────
    try:
        harvest_lines(image, lines, label_text=corrected_text)
    except Exception:
        logger.warning("Could not harvest feedback samples (non-fatal).", exc_info=True)

    # ── Build and persist structured response ─────────────────────────────────
    resp = response_json(
        raw_text=raw_text,
        crnn_text=crnn_text,
        corrected_text=corrected_text,
        image_path=img_path,
        txt_path=txt_path,
        json_path=json_path
    )

    save_json(json_path, resp)

    # Update in-memory latest result (single-worker only; not multi-user safe)
    update_result(resp)

    return resp
