import os
import torch
import clip
import numpy as np
import cv2

from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image

# Flask setup
app = Flask(__name__)

# folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# IMPORTANT: Render için CPU kullan
device = "cpu"

print("Loading CLIP model...")

# jit=False RAM kullanımını düşürür
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

model.eval()

print("CLIP loaded successfully")


# homepage
@app.route("/")
def index():
    return render_template("index.html")


# analyze endpoint
@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "file" not in request.files:
            return jsonify({"message": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"message": "Empty filename"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(filepath)

        print("Saved:", filepath)

        # open image
        image = Image.open(filepath).convert("RGB")

        image_input = preprocess(image).unsqueeze(0).to(device)

        # CLIP inference
        with torch.no_grad():
            features = model.encode_image(image_input)

        print("CLIP inference done")

        # normalize
        features = features / features.norm(dim=-1, keepdim=True)

        feature_value = features.cpu().numpy()[0][0]

        print("Feature value:", feature_value)

        # visualization
        original = cv2.imread(filepath)

        heat = np.ones_like(original) * int(feature_value * 255)

        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)

        output_path = os.path.join(OUTPUT_FOLDER, file.filename)

        cv2.imwrite(output_path, overlay)

        print("Saved visualization:", output_path)

        return send_file(output_path, mimetype="image/jpeg")

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({"message": str(e)})


# health check
@app.route("/health")
def health():
    return "OK"


# local run
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5555))

    app.run(
        host="0.0.0.0",
        port=port
    )