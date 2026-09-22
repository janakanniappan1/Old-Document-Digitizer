import os
import json
import cv2
import tensorflow as tf

keras = tf.keras
layers = keras.layers


class CRNNService:

    def __init__(self):

        self.model = None
        self.char_to_num = None
        self.num_to_char = None

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(
            BASE_DIR,
            "model",
            "crnn_ctc.keras"
        )

        VOCAB_PATH = os.path.join(
            BASE_DIR,
            "model",
            "vocab.json"
        )

        try:

            print("Loading CRNN Model...")

            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab = json.load(f)["vocab"]

            self.char_to_num = layers.StringLookup(
                vocabulary=vocab,
                mask_token=None
            )

            self.num_to_char = layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                mask_token=None,
                invert=True
            )

            self.model = keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

            print("CRNN Loaded Successfully")

        except Exception as e:

            print("CRNN could not be loaded.")
            print(e)

            self.model = None

    # ===========================================
    # Check if model loaded
    # ===========================================

    def is_loaded(self):

        return self.model is not None

    # ===========================================
    # Decode Prediction
    # ===========================================

    def decode_prediction(self, prediction):

        input_len = tf.ones(
            prediction.shape[0]
        ) * prediction.shape[1]

        decoded = keras.backend.ctc_decode(
            prediction,
            input_length=input_len,
            greedy=True
        )[0][0]

        text = []

        for sequence in decoded:

            chars = tf.strings.reduce_join(
                self.num_to_char(sequence)
            ).numpy().decode("utf-8")

            chars = chars.replace("[UNK]", "")

            text.append(chars)

        return text

    # ===========================================
    # Predict Image
    # ===========================================

    def predict(self, image):

        if self.model is None or image is None:
            return ""

        try:
            # OpenCV images are BGR, convert to RGB for tensorflow
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = tf.image.resize(
                image,
                (32, 256)
            )

            image = tf.image.rgb_to_grayscale(image)

            image = tf.cast(
                image,
                tf.float32
            ) / 255.0

            image = tf.expand_dims(
                image,
                axis=0
            )

            prediction = self.model.predict(
                image,
                verbose=0
            )

            text = self.decode_prediction(
                prediction
            )

            if len(text) == 0:
                return ""

            raw_pred = text[0]
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                raw_pred = raw_pred.replace(f"{c}_caps", c)

            return raw_pred

        except Exception as e:

            print("CRNN Prediction Error:", e)

            return ""

    # ===========================================
    # Correct OCR Text
    # ===========================================

    def correct_text(self, paddle_text):

        """
        This method is intentionally simple.

        We let the LLM perform the final correction.

        CRNN is used only when available.
        """

        if paddle_text is None:
            return ""

        return paddle_text.strip()