import os
import torch
import clip
import numpy as np
import cv2

from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static"
OUTPUT_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading CLIP...")

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

print("CLIP loaded successfully")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "file" not in request.files:
            return jsonify({"message": "No file uploaded"})

        file = request.files["file"]

        image = Image.open(file).convert("RGB")

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image_input)

        features_np = features.cpu().numpy()

        norm = np.linalg.norm(features_np)

        image_np = np.array(image)

        h, w, _ = image_np.shape

        center = (w//2, h//2)

        cv2.circle(image_np, center, 50, (0,255,0), 3)

        output_path = "static/output.png"

        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        return jsonify({
            "message": f"Analysis complete | feature_norm={norm:.2f}",
            "output": "/static/output.png"
        })

    except Exception as e:

        print(e)

        return jsonify({"message": str(e)})


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5555))

    app.run(
        host="0.0.0.0",
        port=port
    )
