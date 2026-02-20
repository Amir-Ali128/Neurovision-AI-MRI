from neuro.brain_overlay import overlay_on_brain
from flask import Flask, request, jsonify, render_template
import os

from neuro.yolo import detect
from neuro.clip_model import encode_image
from neuro.activation import ActivationExtractor
from neuro.gradcam import GradCAM
from neuro.cortex_mapper import map_to_brain_region
from neuro.brain_overlay import overlay_on_brain

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

extractor = ActivationExtractor()
gradcam = GradCAM()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(path)

    detections, _ = detect(path)

    embedding = encode_image(path)

    activations = extractor.run(path)

    heatmap_path = gradcam.generate(path)

    brain_result = overlay_on_brain(heatmap_path)

    mapped = []

    for a in activations:

        mapped.append({
            "neuron": a["neuron"],
            "activation": float(a["activation"]),
            "region": map_to_brain_region(a["neuron"])
        })

    return jsonify({
        "detections": detections,
        "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
        "activations": mapped,
        "brain_map": "/" + brain_result
    })

if __name__ == "__main__":
    app.run()
