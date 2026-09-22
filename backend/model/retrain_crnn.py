"""
CRNN Active Learning & Retraining Script
Fine-tunes the existing CRNN model using newly harvested feedback samples and base dataset.
"""

import os
import csv
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

keras = tf.keras
layers = tf.keras.layers

# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "crnn_ctc.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.png")

FEEDBACK_DIR = os.path.join(MODEL_DIR, "feedback_data")
FEEDBACK_CROPS = os.path.join(FEEDBACK_DIR, "crops")
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "labels.csv")

BASE_DATASET_DIR = r"C:\char_data_set"
BASE_IMAGES_DIR = os.path.join(BASE_DATASET_DIR, "augmented_images", "augmented_images1")
BASE_CSV_PATH = os.path.join(BASE_DATASET_DIR, "image_labels.csv")

IMG_H = 32
IMG_W = 256


# =========================
# CTC LOSS LAYER
# =========================
class CTCLossLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length_loss = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length_loss = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length_loss, label_length_loss)
        self.add_loss(loss)

        # Calculate CTC accuracy
        input_length_acc = tf.ones([tf.shape(y_pred)[0]], dtype=tf.int32) * tf.cast(tf.shape(y_pred)[1], tf.int32)
        decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_length_acc, greedy=True)
        pred_dense = tf.cast(decoded[0], tf.int64)

        pred_dense = tf.where(pred_dense == -1, tf.cast(0, tf.int64), pred_dense)
        y_true_cast = tf.cast(y_true, tf.int64)

        max_len = tf.maximum(tf.shape(y_true_cast)[1], tf.shape(pred_dense)[1])
        y_true_padded = tf.pad(y_true_cast, [[0, 0], [0, max_len - tf.shape(y_true_cast)[1]]], constant_values=0)
        pred_padded = tf.pad(pred_dense, [[0, 0], [0, max_len - tf.shape(pred_dense)[1]]], constant_values=0)

        matches = tf.reduce_all(tf.equal(y_true_padded, pred_padded), axis=1)
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))
        self.add_metric(acc, name="accuracy", aggregation="mean")

        return y_pred


def build_crnn(vocab_size):
    image = layers.Input(shape=(IMG_H, IMG_W, 1), name="image", dtype="float32")
    labels = layers.Input(shape=(None,), name="label", dtype="int64")

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(image)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Permute((2, 1, 3))(x)
    new_shape = (-1, x.shape[2] * x.shape[3])
    x = layers.Reshape(target_shape=new_shape)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.Dense(vocab_size + 1, activation="softmax", name="logits")(x)

    output = CTCLossLayer(name="ctc_loss")(labels, x)

    train_model = keras.Model(inputs=[image, labels], outputs=output, name="crnn_ctc")
    inference_model = keras.Model(inputs=image, outputs=x, name="crnn_inference")

    return train_model, inference_model


def load_dataset(max_base_samples=2000):
    image_paths = []
    labels = []

    # 1. Load newly harvested feedback samples
    if os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                crop_file = os.path.join(FEEDBACK_CROPS, r["filename"])
                label = r.get("label", "").strip()
                if os.path.exists(crop_file) and label:
                    image_paths.append(crop_file)
                    labels.append(label)

    print(f"Loaded {len(image_paths)} harvested feedback samples.")

    # 2. Add sample subset from base dataset to prevent catastrophic forgetting
    if os.path.exists(BASE_CSV_PATH) and max_base_samples > 0:
        base_paths = []
        base_labels = []
        with open(BASE_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                full_path = os.path.join(BASE_IMAGES_DIR, os.path.normpath(r["filename"]))
                if os.path.exists(full_path):
                    base_paths.append(full_path)
                    base_labels.append(r["label"])

        if base_paths:
            indices = np.random.choice(len(base_paths), min(len(base_paths), max_base_samples), replace=False)
            for i in indices:
                image_paths.append(base_paths[i])
                labels.append(base_labels[i])
            print(f"Added {len(indices)} base dataset samples for balance.")

    return image_paths, labels


def retrain(epochs=5, batch_size=32, lr=1e-4, max_base_samples=1000):
    print("--- Starting CRNN Retraining & Fine-Tuning ---")

    # Load Vocab
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]

    char_to_num = layers.StringLookup(vocabulary=vocab, mask_token=None)
    vocab_size = len(char_to_num.get_vocabulary())
    print(f"Current Vocab Size: {vocab_size}")

    image_paths, labels = load_dataset(max_base_samples=max_base_samples)
    if not image_paths:
        print("Error: No training samples found. Process documents first to collect feedback crops.")
        return False

    # Shuffle
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    image_paths = np.array(image_paths)[indices].tolist()
    labels = np.array(labels)[indices].tolist()

    split_idx = max(1, int(len(image_paths) * 0.85))
    train_paths, val_paths = image_paths[:split_idx], image_paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")

    def encode_single_sample(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_H, IMG_W])
        lbl = tf.strings.unicode_split(label, input_encoding="UTF-8")
        lbl = char_to_num(lbl)
        return {"image": img, "label": lbl}

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(
            batch_size,
            padded_shapes={"image": [IMG_H, IMG_W, 1], "label": [None]},
            padding_values={"image": 0.0, "label": tf.cast(0, tf.int64)}
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(
            batch_size,
            padded_shapes={"image": [IMG_H, IMG_W, 1], "label": [None]},
            padding_values={"image": 0.0, "label": tf.cast(0, tf.int64)}
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build and initialize model
    train_model, inference_model = build_crnn(vocab_size)

    # Load existing trained weights if present
    if os.path.exists(MODEL_PATH):
        try:
            print("Transferring existing model weights for fine-tuning...")
            existing_model = keras.models.load_model(MODEL_PATH, compile=False)
            inference_model.set_weights(existing_model.get_weights())
            print("Successfully loaded existing weights!")
        except Exception as e:
            print("Notice: Could not directly load existing weights, training with fresh initialization:", e)

    train_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    class SaveInferenceModelCallback(keras.callbacks.Callback):
        def __init__(self, inf_model, filepath):
            super().__init__()
            self.inf_model = inf_model
            self.filepath = filepath
            self.best_loss = np.inf

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss") or logs.get("loss")
            if val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                print(f"\nModel improved (loss: {val_loss:.4f}). Saving to {self.filepath}")
                self.inf_model.save(self.filepath)

    callbacks = [
        SaveInferenceModelCallback(inference_model, MODEL_PATH),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    print("Training started...")
    history = train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model if not saved yet
    if not os.path.exists(MODEL_PATH):
        inference_model.save(MODEL_PATH)

    print(f"Retraining complete! Upgraded model saved at {MODEL_PATH}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRNN Retraining & Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--base_samples", type=int, default=500, help="Base dataset samples")
    args = parser.parse_args()

    retrain(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_base_samples=args.base_samples
    )
