import glob
import hashlib
import io
import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
ALLOWED = {"jpg", "jpeg", "png", "webp"}
THRESHOLD = 0.60  # isse kam confidence par "uncertain" warning

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

TIPS = {
    "cotton": "Recycle into rags, insulation, or new cotton fibre. Keep it clean and dry.",
    "denim": "Send to denim take-back programs, or shred into insulation and new yarn.",
    "polyester": "Can be mechanically or chemically recycled into new polyester fibre. Do not burn.",
    "silk": "Rarely recycled at scale. Best reused, repaired, or donated; scraps suit crafts.",
    "wool": "Recycle into felt, blankets, or insulation. Wool is also compostable if untreated.",
}
INFO = {
    "cotton": "Natural plant fibre. Soft, breathable, absorbent.",
    "denim": "Sturdy twill cotton weave, usually indigo dyed.",
    "polyester": "Synthetic fibre from petroleum. Durable, quick drying.",
    "silk": "Natural protein fibre with a smooth, glossy surface.",
    "wool": "Natural animal fibre. Warm, elastic, often fuzzy.",
}
# Home page par dikhne wale numbers. Retrain ke baad yahan update kar dena.
STATS = {"classes": 5, "images": "2,692", "accuracy": "66%", "realworld": "n/a"}
DIST = {"cotton": 708, "wool": 659, "silk": 607, "polyester": 371, "denim": 379}

try:
    with open(os.path.join(MODEL_DIR, "class_names.json"), encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
except Exception:
    CLASS_NAMES = ["cotton", "denim", "polyester", "silk", "wool"]  # alphabetical, training jaisa

_lock = threading.Lock()


def _first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(MODEL_DIR, pat)))
        if hits:
            return hits[0]
    return None


def load_predictor():
    """Returns (predict_fn or None, status text). predict_fn: float32 array (1,224,224,3) -> probs."""
    files = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else None
    if files is None:
        return None, f"Folder not found: {MODEL_DIR}"
    keras_path = _first(["ecotex_model*.keras", "*.keras", "*.h5"])
    tflite_path = _first(["*.tflite"])
    if not keras_path and not tflite_path:
        return None, f"No .keras or .tflite file in {MODEL_DIR}. Files found: {files}"
    print("Loading TensorFlow... pehli baar 2-5 minute lag sakte hain. Ctrl+C mat dabana, intezar karo.", flush=True)
    try:
        import tensorflow as tf
    except Exception as e:  # noqa: BLE001
        return None, f"TensorFlow is not installed ({e}). Run: python -m pip install tensorflow"

    errors = []
    if keras_path:
        try:
            m = tf.keras.models.load_model(keras_path, compile=False)
            return (lambda a: m.predict(a, verbose=0)[0]), f"Loaded {os.path.basename(keras_path)}"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{os.path.basename(keras_path)} failed: {e}")
    if tflite_path:
        try:
            it = tf.lite.Interpreter(model_path=tflite_path)
            it.allocate_tensors()
            i_in, i_out = it.get_input_details()[0], it.get_output_details()[0]

            def run(a):
                it.set_tensor(i_in["index"], a.astype(i_in["dtype"]))
                it.invoke()
                return it.get_tensor(i_out["index"])[0]

            return run, f"Loaded {os.path.basename(tflite_path)} (TFLite)"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{os.path.basename(tflite_path)} failed: {e}")
    return None, " | ".join(errors)


predictor, MODEL_STATUS = load_predictor()
print("=" * 60)
print("Model folder:", MODEL_DIR)
print("MODEL STATUS:", MODEL_STATUS)
print("MODE:", "REAL MODEL" if predictor else "DEMO (placeholder results)")
print("=" * 60)


def predict(img_bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    img = img.resize((224, 224))  # training jaisa resize; normalization model ke andar hai
    if predictor is not None:
        arr = np.expand_dims(np.asarray(img, dtype="float32"), 0)  # 0-255 values
        with _lock:
            probs = np.asarray(predictor(arr), dtype="float64")
        return probs, False
    seed = int(hashlib.md5(img_bytes).hexdigest()[:8], 16)  # demo: same image = same result
    return np.random.default_rng(seed).dirichlet(np.ones(len(CLASS_NAMES)) * 0.6), True


@app.route("/")
def home():
    return render_template("home.html", page="home", classes=CLASS_NAMES, stats=STATS, dist=DIST)


@app.route("/about")
def about():
    return render_template("about.html", page="about", classes=CLASS_NAMES, info=INFO)


@app.route("/classify")
def classify():
    return render_template("classify.html", page="classify", demo=predictor is None, status=MODEL_STATUS)


@app.route("/health")
def health():
    return jsonify(status="ok", demo=predictor is None, model_status=MODEL_STATUS, classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict_route():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="No image received. Please choose or capture a photo."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "jpg"
    if ext not in ALLOWED:
        return jsonify(error="Unsupported file type. Use JPG, PNG or WEBP."), 400
    try:
        probs, demo = predict(f.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="This file is not a valid image."), 400
    names = CLASS_NAMES if len(CLASS_NAMES) == len(probs) else [f"class_{i}" for i in range(len(probs))]
    order = np.argsort(probs)[::-1][:3]
    top3 = [{"label": names[i], "confidence": float(probs[i])} for i in order]
    best = top3[0]
    return jsonify(
        label=best["label"], confidence=best["confidence"], top3=top3,
        tip=TIPS.get(best["label"], ""), info=INFO.get(best["label"], ""),
        uncertain=best["confidence"] < THRESHOLD, demo=demo,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 8 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
