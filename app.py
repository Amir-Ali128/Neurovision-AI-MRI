import os
import threading

import clip
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, UnidentifiedImageError

# Flask
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_UPLOAD_MB", "15")
) * 1024 * 1024

# static klasörü garanti olsun
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Render için CPU kullan (GPU detection RAM artırabilir)
device = "cpu"
print("Using device:", device)

# CLIP singleton state
_model = None
_preprocess = None
_model_lock = threading.Lock()
_model_error = None



def get_clip_model():
    """Thread-safe lazy singleton for CLIP model."""
    global _model, _preprocess, _model_error

    if _model is not None and _preprocess is not None:
        return _model, _preprocess

    with _model_lock:
        if _model is None or _preprocess is None:
            try:
                print("Loading CLIP...")
                _model, _preprocess = clip.load("ViT-B/32", device=device, jit=False)
                _model.eval()
                _model_error = None
                print("CLIP loaded successfully")
            except Exception as exc:
                _model_error = str(exc)
                raise

    return _model, _preprocess



def warmup_model():
    """Optional startup warmup to avoid first-request cold start timeout."""
    try:
        get_clip_model()
        return True
    except Exception as exc:
        print("Model warmup failed:", exc)
        return False


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Health check (process alive)
@app.route("/health")
def health():
    return "OK"


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_err):
    return jsonify({"message": "File too large"}), 413


# Readiness check (model loaded)
@app.route("/ready")
def ready():
    if _model is None:
        return jsonify({"ready": False, "message": "Model not loaded yet"}), 503

    return jsonify({"ready": True}), 200


# Analyze endpoint
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # file kontrol
        if "file" not in request.files:
            return jsonify({"message": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"message": "Empty filename"}), 400

        # RAM üzerinden oku (disk kullanma)
        image = Image.open(file).convert("RGB")

        model, preprocess = get_clip_model()
        image_input = preprocess(image).unsqueeze(0).to(device)

        # CLIP inference
        with torch.no_grad():
            features = model.encode_image(image_input)

        features_np = features.cpu().numpy()
        norm = float(np.linalg.norm(features_np))

        # Visualization
        image_np = np.array(image)
        h, w, _ = image_np.shape
        center = (w // 2, h // 2)
        cv2.circle(image_np, center, 50, (0, 255, 0), 3)

        # Save output
        output_path = os.path.join(STATIC_FOLDER, "output.png")
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        return jsonify(
            {
                "message": f"Analysis complete | feature_norm={norm:.2f}",
                "output": "/static/output.png",
            }
        )

    except UnidentifiedImageError:
        return jsonify({"message": "Invalid image file"}), 400
    except Exception as e:
        print("ERROR:", e)

        return jsonify({"message": str(e)}), 500


# Optional eager preload for Gunicorn --preload usage
if os.environ.get("CLIP_PRELOAD_ON_START", "0") == "1":
    warmup_model()


# Local run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5555))

    if os.environ.get("CLIP_PRELOAD_ON_START", "0") == "1":
        warmup_model()

    app.run(host="0.0.0.0", port=port, debug=False)
