import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import gc
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import clip

app = Flask(__name__)

device = "cpu"
torch.set_num_threads(1)  # RAM/CPU patlamasını azaltır

# Smaller CLIP model = less RAM
model, preprocess = clip.load("RN50", device=device)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        # Read without saving to disk
        image = Image.open(BytesIO(f.read())).convert("RGB")

        # Downscale to reduce RAM
        image.thumbnail((768, 768))

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = model.encode_image(image_input)

        # cleanup
        del image_input, feats, image
        gc.collect()

        return jsonify({"status": "success", "message": "Analysis complete"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health():
    return "OK", 200
