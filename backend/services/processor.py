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

    # ── Generate unified, unique base filename ────────────────────────────────
    base_name = generate_base_name()
    img_name = f"{base_name}.jpg"
    txt_name = f"{base_name}.txt"
    json_name = f"{base_name}.json"

    img_path = os.path.join(OUTPUT_FOLDER, img_name)
    txt_path = os.path.join(OUTPUT_FOLDER, txt_name)
    json_path = os.path.join(OUTPUT_FOLDER, json_name)

    raw_text = ""
    crnn_text = ""
    corrected_text = ""
    lines = []
    output_img = image

    # ── 1. Fast Cloud Multimodal AI Vision Path ──────────────────────────────
    # If Gemini Cloud AI is configured and AI mode is requested, transcribe directly
    # via Google Cloud in 2-4 seconds with zero local CPU bottleneck or memory usage.
    if use_llm and hasattr(llm, "has_vision") and llm.has_vision():
        try:
            cloud_transcription = llm.transcribe_image(image)
            if cloud_transcription and cloud_transcription.strip():
                raw_text = cloud_transcription
                crnn_text = cloud_transcription
                corrected_text = cloud_transcription
        except Exception as e:
            logger.warning("Cloud vision transcription fallback: %s", e)

    # ── 2. Local OCR Fallback / Fast Mode ────────────────────────────────────
    # Runs when Cloud Vision is unavailable or user explicitly chose 'fast' mode
    if not raw_text:
        result = paddle_ocr.extract_text(image)
        raw_text = result["text"]
        lines = result["lines"]
        output_img = paddle_ocr.draw_boxes(image, lines)

        crnn_text = crnn.correct_text(raw_text)

        if use_llm:
            corrected_text = llm.correct_text(crnn_text)
        else:
            corrected_text = crnn_text

    # ── 3. Save annotated or processed image ─────────────────────────────────
    cv2.imwrite(img_path, output_img)

    # ── 4. Save text output ──────────────────────────────────────────────────
    text_content = f"--- Original OCR ---\n{raw_text}\n\n--- AI Corrected ---\n{corrected_text}"
    save_text(txt_path, text_content)

    # ── 5. Harvest feedback crops (if bounding boxes available) ──────────────
    if lines:
        try:
            harvest_lines(image, lines, label_text=corrected_text)
        except Exception:
            logger.warning("Could not harvest feedback samples (non-fatal).", exc_info=True)

    # ── 6. Build and persist structured response ─────────────────────────────
    resp = response_json(
        raw_text=raw_text,
        crnn_text=crnn_text,
        corrected_text=corrected_text,
        image_path=img_path,
        txt_path=txt_path,
        json_path=json_path
    )

    save_json(json_path, resp)

    # Update in-memory latest result
    update_result(resp)

    return resp
