import os
import torch
import clip
import numpy as np
import cv2

from flask import Flask, request, render_template, jsonify
from PIL import Image

# Flask
app = Flask(__name__)

# static klasörü garanti olsun
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Render için CPU kullan (GPU detection RAM artırabilir)
device = "cpu"
print("Using device:", device)

# Load CLIP once
print("Loading CLIP...")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()
print("CLIP loaded successfully")


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Analyze endpoint
@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # file kontrol
        if "file" not in request.files:
            return jsonify({"message": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"message": "Empty filename"})


        # RAM üzerinden oku (disk kullanma)
        image = Image.open(file).convert("RGB")

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

        cv2.circle(image_np, center, 50, (0,255,0), 3)

        # Save output
        output_path = os.path.join(STATIC_FOLDER, "output.png")

        cv2.imwrite(
            output_path,
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        )

        return jsonify({
            "message": f"Analysis complete | feature_norm={norm:.2f}",
            "output": "/static/output.png"
        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "message": str(e)
        })


# Health check
@app.route("/health")
def health():
    return "OK"


# Local run
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5555))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )