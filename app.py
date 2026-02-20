
from flask import Flask, request, jsonify, render_template
import os
import numpy as np

# Neuro modules
from neuro.yolo import detect
from neuro.clip_model import encode_image
from neuro.activation import ActivationExtractor
from neuro.gradcam import GradCAM
from neuro.cortex_mapper import map_to_brain_region
from neuro.visualize import overlay_heatmap


# Flask setup
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# Load models once (important for performance)
extractor = ActivationExtractor()
gradcam = GradCAM()


# HOME PAGE
@app.route("/")
def index():
    return render_template("index.html")


# ANALYZE ENDPOINT
@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:

        # YOLO detection
        detections, _ = detect(path)

        # CLIP embedding
        embedding = encode_image(path)

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # Neural activations
        activations = extractor.run(path)

        mapped = []

        for a in activations:

            mapped.append({
                "neuron": int(a["neuron"]),
                "activation": float(a["activation"]),
                "region": map_to_brain_region(a["neuron"])
            })

        # GradCAM heatmap
        heatmap = gradcam.generate(path)

        heatmap_array = np.array(heatmap)

        output_path = overlay_heatmap(
            path,
            heatmap_array,
            output_name="result.jpg"
        )

        return jsonify({

            "detections": detections,
            "embedding": embedding,
            "activations": mapped,
            "overlay_image": "/" + output_path

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Render requires this
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
