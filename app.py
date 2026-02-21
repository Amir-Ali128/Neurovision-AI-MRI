import os
import torch
import clip
import numpy as np
import cv2

from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

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

        file = request.files["file"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(filepath)

        image = Image.open(filepath).convert("RGB")

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image_input)

        features = features / features.norm(dim=-1, keepdim=True)

        activation = features.cpu().numpy()[0]

        activation_strength = np.mean(activation)

        print("Activation:", activation_strength)

        # LOAD REAL BRAIN TEMPLATE
        brain = cv2.imread("static/brain_template.png")

        h, w, _ = brain.shape

        overlay = brain.copy()

        # create activation circle
        center = (
            int(w * np.random.uniform(0.3, 0.7)),
            int(h * np.random.uniform(0.3, 0.7))
        )

        radius = int(activation_strength * 300)

        heat_color = (
            0,
            int(activation_strength * 255),
            255
        )

        cv2.circle(
            overlay,
            center,
            radius,
            heat_color,
            -1
        )

        result = cv2.addWeighted(
            brain,
            0.7,
            overlay,
            0.3,
            0
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            file.filename
        )

        cv2.imwrite(output_path, result)

        return send_file(
            output_path,
            mimetype="image/png"
        )

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "message": str(e)
        })


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5555))

    app.run(
        host="0.0.0.0",
        port=port
    )
