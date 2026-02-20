from flask import Flask, request, jsonify, render_template
import os

# Neuro modules
from neuro.yolo import detect
from neuro.clip_model import encode_image
from neuro.activation import ActivationExtractor
from neuro.gradcam import GradCAM
from neuro.cortex_mapper import map_to_brain_region
from neuro.brain_overlay import overlay_on_brain

# Flask setup
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models once
extractor = ActivationExtractor()
gradcam = GradCAM()


# Homepage
@app.route("/")
def index():
    return render_template("index.html")


# MRI Analysis Endpoint
@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # 1️⃣ get uploaded image
        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file uploaded"})

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        print("Image saved:", path)


        # 2️⃣ YOLO detection
        detections, _ = detect(path)
        print("YOLO done")


        # 3️⃣ CLIP embedding
        embedding = encode_image(path)
        print("CLIP done")


        # 4️⃣ Activation extraction
        activations = extractor.run(path)
        print("Activation done")


        # 5️⃣ GradCAM heatmap
        heatmap_path = gradcam.generate(path)
        print("GradCAM heatmap:", heatmap_path)


        # 6️⃣ Activation → brain region mapping
        mapped = []

        for a in activations:

            mapped.append({
                "neuron": int(a["neuron"]),
                "activation": float(a["activation"]),
                "region": map_to_brain_region(a["neuron"])
            })


        # 7️⃣ Anatomik beyin üzerine çiz
        brain_result_path = overlay_on_brain(mapped)

        print("Brain overlay:", brain_result_path)


        # 8️⃣ response
        return jsonify({

            "success": True,

            "detections": detections,

            "embedding": embedding,

            "activations": mapped,

            "brain_image":  brain_result_path
        })


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({

            "success": False,

            "error": str(e)
        })


# Render production server
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
