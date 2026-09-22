import cv2
import numpy as np
from paddleocr import PaddleOCR

from services.image_service import ImageService


class PaddleOCRService:

    def __init__(self):

        self.image_service = ImageService()

        print("Loading PaddleOCR Model...")

        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_textline_orientation=True,
            use_space_char=True,
            show_log=False,
            det_db_thresh=0.10,
            det_db_box_thresh=0.30,
            det_db_unclip_ratio=2.0,
            det_limit_side_len=1280
        )

        print("PaddleOCR Loaded Successfully")

    # =====================================
    # OCR ON SINGLE IMAGE
    # =====================================
    def ocr_image(self, image):
        if image is None:
            return []

        h, w = image.shape[:2]
        max_dim = max(h, w)
        downscale_ratio = 1.0

        target_img = image
        if max_dim > 1600:
            downscale_ratio = 1600.0 / max_dim
            new_w = int(w * downscale_ratio)
            new_h = int(h * downscale_ratio)
            target_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        result = self.ocr.ocr(target_img, cls=True)

        lines = []

        if isinstance(result, list):

            if len(result) > 0:

                data = result[0]

                if data is None:
                    return []

                for item in data:

                    try:

                        bbox = item[0]

                        text = item[1][0]

                        confidence = float(item[1][1])

                        if confidence > 0.10 and text.strip():

                            if downscale_ratio != 1.0:
                                bbox = [[p[0] / downscale_ratio, p[1] / downscale_ratio] for p in bbox]

                            lines.append({

                                "text": text.strip(),

                                "confidence": confidence,

                                "bbox": bbox

                            })

                    except Exception:
                        pass

        return lines

    # =====================================
    # IOU
    # =====================================
    def bbox_iou(self, box1, box2):

        x1 = max(min(p[0] for p in box1), min(p[0] for p in box2))
        y1 = max(min(p[1] for p in box1), min(p[1] for p in box2))
        x2 = min(max(p[0] for p in box1), max(p[0] for p in box2))
        y2 = min(max(p[1] for p in box1), max(p[1] for p in box2))

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        if inter == 0:
            return 0

        area1 = (
            (max(p[0] for p in box1) - min(p[0] for p in box1))
            *
            (max(p[1] for p in box1) - min(p[1] for p in box1))
        )

        area2 = (
            (max(p[0] for p in box2) - min(p[0] for p in box2))
            *
            (max(p[1] for p in box2) - min(p[1] for p in box2))
        )

        return inter / (area1 + area2 - inter + 1e-6)

    # =====================================
    # REMOVE DUPLICATES
    # =====================================
    def remove_duplicates(self, lines):
        kept = []
        for line in sorted(lines, key=lambda x: (-len(x["text"]), -x["confidence"])):
            duplicate = False
            for k in kept:
                x1 = max(min(p[0] for p in line["bbox"]), min(p[0] for p in k["bbox"]))
                y1 = max(min(p[1] for p in line["bbox"]), min(p[1] for p in k["bbox"]))
                x2 = min(max(p[0] for p in line["bbox"]), max(p[0] for p in k["bbox"]))
                y2 = min(max(p[1] for p in line["bbox"]), max(p[1] for p in k["bbox"]))

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                
                area1 = (max(p[0] for p in line["bbox"]) - min(p[0] for p in line["bbox"])) * (max(p[1] for p in line["bbox"]) - min(p[1] for p in line["bbox"]))
                area2 = (max(p[0] for p in k["bbox"]) - min(p[0] for p in k["bbox"])) * (max(p[1] for p in k["bbox"]) - min(p[1] for p in k["bbox"]))

                min_area = min(area1, area2)
                
                # Raised threshold to 0.65 — only remove truly duplicate/contained boxes,
                # not just nearby ones (important for slanted/angled handwriting)
                if min_area > 0 and (inter / min_area) > 0.65:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(line)
        return kept

    # =====================================
    # GROUP AND SORT TEXT
    # =====================================
    def group_and_sort_text(self, lines):
        if not lines:
            return ""
            
        lines.sort(key=lambda x: min(p[1] for p in x["bbox"]))
        
        grouped = []
        current_line = [lines[0]]
        
        for line in lines[1:]:
            avg_y1_min = sum(min(p[1] for p in item["bbox"]) for item in current_line) / len(current_line)
            avg_y1_max = sum(max(p[1] for p in item["bbox"]) for item in current_line) / len(current_line)
            
            y2_min = min(p[1] for p in line["bbox"])
            y2_max = max(p[1] for p in line["bbox"])
            
            overlap = max(0, min(avg_y1_max, y2_max) - max(avg_y1_min, y2_min))
            min_height = min(avg_y1_max - avg_y1_min, y2_max - y2_min)
            
            # Lowered from 0.4 → 0.25: only group boxes on the same row.
            # Higher threshold incorrectly merges lines in slanted/angled handwriting.
            if min_height > 0 and (overlap / min_height) > 0.25:
                current_line.append(line)
            else:
                grouped.append(current_line)
                current_line = [line]
                
        grouped.append(current_line)
        
        final_text = []
        for group in grouped:
            group.sort(key=lambda x: min(p[0] for p in x["bbox"]))
            final_text.append(" ".join(item["text"] for item in group))
            
        return "\n".join(final_text)

    # =====================================
    # DRAW BOXES
    # =====================================
    def draw_boxes(self, image, lines):

        output = image.copy()

        for line in lines:

            pts = np.array(
                [[p[0], p[1]] for p in line["bbox"]],
                dtype=np.int32
            )

            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(
                output,
                [pts],
                True,
                (0, 255, 0),
                2
            )

        return output

    # =====================================
    # COMPLETE OCR
    # =====================================
    def extract_text(self, image):
        # Fast primary pass
        lines = self.ocr_image(image)

        # Adaptive fallback: only if no text lines were detected at normal scale, run resized pass
        if len(lines) == 0 and hasattr(self.image_service, 'preprocess'):
            try:
                processed = self.image_service.preprocess(image)
                proc_lines = self.ocr_image(processed)
                scale = getattr(self.image_service, 'scale', 2.0)
                if scale and scale != 1.0:
                    for item in proc_lines:
                        item["bbox"] = [[p[0] / scale, p[1] / scale] for p in item["bbox"]]
                lines.extend(proc_lines)
            except Exception as e:
                print("Warning: Adaptive OCR fallback error:", e)

        lines = self.remove_duplicates(lines)
        full_text = self.group_and_sort_text(lines)

        return {
            "text": full_text,
            "lines": lines
        }