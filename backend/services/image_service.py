import cv2
import numpy as np


class ImageService:

    def __init__(self):
        self.scale = 1.0

    # ==========================================
    # Resize Image
    # ==========================================
    def resize(self, image):
        h, w = image.shape[:2]

        image = cv2.resize(
            image,
            (int(w * self.scale), int(h * self.scale)),
            interpolation=cv2.INTER_CUBIC
        )

        return image

    # ==========================================
    # CLAHE Contrast Enhancement
    # ==========================================
    def clahe(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ==========================================
    # Sharpen Image
    # ==========================================
    def sharpen(self, image):

        blur = cv2.GaussianBlur(image, (0, 0), 3)

        sharp = cv2.addWeighted(
            image,
            1.8,
            blur,
            -0.8,
            0
        )

        return sharp

    # ==========================================
    # Remove Noise (Toned down)
    # ==========================================
    def denoise(self, image):

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            5,  # Reduced from 10 to preserve handwriting strokes
            5,  # Reduced from 10
            7,
            21
        )

    # ==========================================
    # Adaptive Threshold
    # ==========================================
    def threshold(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ==========================================
    # Deskew Image
    # ==========================================
    def deskew(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert np.where (row, col) to (x, y) points
        coords = np.column_stack(np.where(gray < 250)[::-1])

        if len(coords) == 0:
            return image

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        if abs(angle) < 0.5:
            return image

        (h, w) = image.shape[:2]

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    # ==========================================
    # Complete Preprocessing Pipeline
    # ==========================================
    def preprocess(self, image):
        # Neural networks usually perform best on natural images without harsh filters.
        # CLAHE, sharpening, and denoising were destroying the pencil strokes of the 'J'.
        image = self.resize(image)

        return image