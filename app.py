from flask import Flask, request, jsonify, render_template
import os

# neuro modules
from neuro.yolo import detect
from neuro.clip_model import encode_image
from neuro.activation import ActivationExtractor
from neuro.gradcam import GradCAM
from neuro.cortex_mapper import map_to_brain_region
from neuro.brain_overlay import overlay_on_brain

# Flask
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🚨 LAZY LOAD (CRITICAL FOR RAILWAY / RENDER)
extractor = None
gradcam = None


# Homepage
@app.route("/")
def index():
    return render_template("index.html")


# Analyze endpoint
@app.route("/analyze", methods=["POST"])
def analyze():

    global extractor, gradcam

    try:

        # load models ONLY when needed
        if extractor is None:
            print("Loading ActivationExtractor...")
            extractor = ActivationExtractor()

        if gradcam is None:
            print("Loading GradCAM...")
            gradcam = GradCAM()


        # get file
        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            })


        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        print("Image saved:", path)


        # YOLO detection
        detections, _ = detect(path)

        print("YOLO done")


        # CLIP embedding
        embedding = encode_image(path)

        print("CLIP done")


        # Activation extraction
        activations = extractor.run(path)

        print("Activation done")


        # GradCAM heatmap
        heatmap_path = gradcam.generate(path)

        print("GradCAM done")


        # map neurons to brain regions
        mapped = []

        for a in activations:

            mapped.append({

                "neuron": int(a["neuron"]),
                "activation": float(a["activation"]),
                "region": map_to_brain_region(a["neuron"])

            })


        # overlay on anatomical brain
        brain_result_path = overlay_on_brain(mapped)

        print("Brain overlay done:", brain_result_path)


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


# production run
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
