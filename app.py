import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image

# CLIP (OpenAI repo)
import clip

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP once
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP loaded successfully ✅")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return "OK"


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # 1) File check
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        # 2) Save
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)
        print("Saved:", filepath)

        # 3) Load image safely
        image = Image.open(filepath).convert("RGB")

        # 4) Preprocess & inference
        image_input = preprocess(image).unsqueeze(0).to(device)

        print("Running CLIP inference...")
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # (optional) convert to list just to show it worked
        feat_norm = torch.norm(image_features, dim=-1).item()

        print("CLIP inference done ✅")

        return jsonify({
            "status": "success",
            "message": "Analysis complete ✅",
            "feature_norm": feat_norm
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # local run (Render uses gunicorn)
    app.run(host="0.0.0.0", port=5555, debug=True)
