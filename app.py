from flask import Flask, request, jsonify, render_template
import os

# Neuro modules
from neuro.yolo import detect
from neuro.clip_model import encode_image
from neuro.activation import ActivationExtractor
from neuro.gradcam import GradCAM
from neuro.cortex_mapper import map_to_brain_region
from neuro.brain_overlay import overlay_on_brain

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy load models
extractor = ActivationExtractor()
gradcam = GradCAM()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded"
            })

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            })

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        print("Saved:", path)

        # YOLO detect
        detections, _ = detect(path)
        print("YOLO done")

        # CLIP embedding
        embedding = encode_image(path)
        print("CLIP done")

        # Activation extraction
        activations = extractor.run(path)

        if activations is None:
            activations = []

        print("Activation done:", len(activations))

        # GradCAM
        heatmap_path = gradcam.generate(path)
        print("GradCAM:", heatmap_path)

        # Map neurons to brain regions
        mapped = []

        for a in activations:

            mapped.append({
                "neuron": int(a.get("neuron", 0)),
                "activation": float(a.get("activation", 0)),
                "region": map_to_brain_region(a.get("neuron", 0))
            })

        # Draw on anatomical brain
        brain_result_path = overlay_on_brain(mapped)

        print("Brain overlay:", brain_result_path)

        return jsonify({

            "success": True,

            "detections": detections,

            "embedding": embedding,

            "activations": mapped,

            "brain_image": "/" + brain_result_path
        })


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
