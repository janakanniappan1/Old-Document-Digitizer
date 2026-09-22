import os
import json
import logging
import cv2

logger = logging.getLogger(__name__)


class CRNNService:

    def __init__(self):
        self.model = None
        self.char_to_num = None
        self.num_to_char = None
        self._tf = None
        self._keras = None
        self._loaded = False

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(BASE_DIR, "model", "crnn_ctc.keras")
        self.VOCAB_PATH = os.path.join(BASE_DIR, "model", "vocab.json")

    def _ensure_loaded(self):
        """Lazy-load TensorFlow and Keras model only when character prediction is explicitly requested."""
        if self._loaded:
            return
        self._loaded = True

        try:
            logger.info("Loading TensorFlow and CRNN model on demand...")
            import tensorflow as tf
            self._tf = tf
            self._keras = tf.keras

            with open(self.VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab = json.load(f)["vocab"]

            layers = self._keras.layers
            self.char_to_num = layers.StringLookup(vocabulary=vocab, mask_token=None)
            self.num_to_char = layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                mask_token=None,
                invert=True
            )

            self.model = self._keras.models.load_model(self.MODEL_PATH, compile=False)
            logger.info("CRNN loaded successfully.")
        except Exception as e:
            logger.warning("CRNN could not be loaded: %s", e)
            self.model = None

    def is_loaded(self):
        return self.model is not None

    def decode_prediction(self, prediction):
        self._ensure_loaded()
        if not self._tf or not self._keras:
            return []

        input_len = self._tf.ones(prediction.shape[0]) * prediction.shape[1]
        decoded = self._keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]

        text = []
        for sequence in decoded:
            chars = self._tf.strings.reduce_join(self.num_to_char(sequence)).numpy().decode("utf-8")
            chars = chars.replace("[UNK]", "")
            text.append(chars)
        return text

    def predict(self, image):
        self._ensure_loaded()
        if self.model is None or image is None or self._tf is None:
            return ""

        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self._tf.image.resize(image, (32, 256))
            image = self._tf.image.rgb_to_grayscale(image)
            image = self._tf.cast(image, self._tf.float32) / 255.0
            image = self._tf.expand_dims(image, axis=0)

            prediction = self.model.predict(image, verbose=0)
            text = self.decode_prediction(prediction)

            if len(text) == 0:
                return ""

            raw_pred = text[0]
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                raw_pred = raw_pred.replace(f"{c}_caps", c)

            return raw_pred
        except Exception as e:
            logger.warning("CRNN Prediction Error: %s", e)
            return ""

    def correct_text(self, paddle_text):
        """Pass-through verification step — LLM performs final linguistic correction."""
        if paddle_text is None:
            return ""
        return paddle_text.strip()