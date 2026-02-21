import os
import torch
import clip
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Flask setup
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model ONCE at startup
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

print("CLIP loaded successfully")

# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Analyze route
@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file
        file.save(filepath)
        print("Saved:", filepath)

        # Load image safely with PIL
        image = Image.open(filepath).convert("RGB")

        # Preprocess for CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        print("Analysis done")

        return jsonify({
            "status": "success",
            "message": "Analysis complete"
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        })


# Health check
@app.route("/health")
def health():
    return "OK"


# Run locally (not used in gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)