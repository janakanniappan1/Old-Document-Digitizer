import os
import csv
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

keras = tf.keras
layers = tf.keras.layers

# =========================
# CONFIG
# =========================
DATASET_DIR = os.environ.get("DATASET_DIR", r"C:\char_data_set")
IMAGES_DIR = os.path.join(DATASET_DIR, "augmented_images", "augmented_images1")
CSV_PATH = os.path.join(DATASET_DIR, "image_labels.csv")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crnn_ctc.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_H = 32
IMG_W = 256
BATCH = 64
EPOCHS = 10

# =========================
# DATA PREPARATION
# =========================

print("Loading dataset paths...")
image_paths = []
labels = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        filename = r["filename"]
        full_path = os.path.join(IMAGES_DIR, os.path.normpath(filename))
        
        label = r["label"]
        
        image_paths.append(full_path)
        labels.append(label)

print(f"Total samples found: {len(image_paths)}")

# Build Vocab
unique_chars = sorted(list(set(labels)))
print(f"Vocab size: {len(unique_chars)}")

# Save Vocab FIRST
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump({"vocab": unique_chars}, f)

char_to_num = layers.StringLookup(vocabulary=unique_chars, mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

VOCAB_SIZE = len(char_to_num.get_vocabulary())

seed = 42
np.random.seed(seed)
indices = np.arange(len(image_paths))
np.random.shuffle(indices)

image_paths = np.array(image_paths)[indices].tolist()
labels = np.array(labels)[indices].tolist()

split_idx = int(len(image_paths) * 0.8)
train_paths = image_paths[:split_idx]
train_labels = labels[:split_idx]
val_paths = image_paths[split_idx:]
val_labels = labels[split_idx:]

print(f"Train samples: {len(train_paths)}")
print(f"Val samples: {len(val_paths)}")

# =========================
# TF DATA PIPELINE
# =========================

def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    
    return {"image": img, "label": label}

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.padded_batch(
    BATCH,
    padded_shapes={"image": [IMG_H, IMG_W, 1], "label": [None]},
    padding_values={"image": 0.0, "label": tf.cast(0, tf.int64)}
).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.padded_batch(
    BATCH,
    padded_shapes={"image": [IMG_H, IMG_W, 1], "label": [None]},
    padding_values={"image": 0.0, "label": tf.cast(0, tf.int64)}
).prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================
# MODEL
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
        
        # Calculate CTC Exact Match Accuracy
        input_length_acc = tf.ones([tf.shape(y_pred)[0]], dtype=tf.int32) * tf.cast(tf.shape(y_pred)[1], tf.int32)
        decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_length_acc, greedy=True)
        pred_dense = tf.cast(decoded[0], tf.int64)
        
        # CTC Decode pads with -1, but our true labels are padded with 0
        pred_dense = tf.where(pred_dense == -1, tf.cast(0, tf.int64), pred_dense)
        y_true_cast = tf.cast(y_true, tf.int64)
        
        # Pad both to the max length in this batch so they match perfectly
        max_len = tf.maximum(tf.shape(y_true_cast)[1], tf.shape(pred_dense)[1])
        y_true_padded = tf.pad(y_true_cast, [[0, 0], [0, max_len - tf.shape(y_true_cast)[1]]], constant_values=0)
        pred_padded = tf.pad(pred_dense, [[0, 0], [0, max_len - tf.shape(pred_dense)[1]]], constant_values=0)
        
        matches = tf.reduce_all(tf.equal(y_true_padded, pred_padded), axis=1)
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))
        self.add_metric(acc, name="accuracy", aggregation="mean")
        
        return y_pred

def build_crnn():
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

    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="logits")(x)

    output = CTCLossLayer(name="ctc_loss")(labels, x)
    
    model = keras.Model(inputs=[image, labels], outputs=output, name="crnn_ctc")
    inference_model = keras.Model(inputs=image, outputs=x, name="crnn_inference")
    
    return model, inference_model

train_model, inference_model = build_crnn()
train_model.compile(optimizer=keras.optimizers.Adam(1e-3))
train_model.summary()

class SaveInferenceModelCallback(keras.callbacks.Callback):
    def __init__(self, inference_model, filepath):
        super().__init__()
        self.inference_model = inference_model
        self.filepath = filepath
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            print(f"\nSaving inference model to {self.filepath}")
            self.inference_model.save(self.filepath)

class PlotMetricsCallback(keras.callbacks.Callback):
    def __init__(self, save_path="model/training_metrics.png"):
        super().__init__()
        self.save_path = save_path
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["accuracy"].append(logs.get("accuracy", 0))
        self.history["val_accuracy"].append(logs.get("val_accuracy", 0))
        
        epochs = range(1, len(self.history["loss"]) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["loss"], label="Train Loss (CTC)", marker='o')
        plt.plot(epochs, self.history["val_loss"], label="Val Loss (CTC)", marker='o')
        plt.title("CTC Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["accuracy"], label="Train Accuracy", marker='o')
        plt.plot(epochs, self.history["val_accuracy"], label="Val Accuracy", marker='o')
        plt.title("Sequence Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        print(f"\nSaved training graph to {self.save_path}")

callbacks = [
    SaveInferenceModelCallback(inference_model, MODEL_PATH),
    PlotMetricsCallback(os.path.join(MODEL_DIR, "training_metrics.png")),
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
]

print("Starting training...")
train_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"Training complete. Inference model saved at {MODEL_PATH}")