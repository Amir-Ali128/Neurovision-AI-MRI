from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# lazy globals
detect = None
encode_image = None
ActivationExtractor = None
GradCAM = None
map_to_brain_region = None
overlay_on_brain = None

extractor = None
gradcam = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    global detect, encode_image, ActivationExtractor, GradCAM
    global map_to_brain_region, overlay_on_brain
    global extractor, gradcam

    try:

        # LAZY IMPORT
        if detect is None:

            print("Loading neuro modules...")

            from neuro.yolo import detect as d
            from neuro.clip_model import encode_image as e
            from neuro.activation import ActivationExtractor as A
            from neuro.gradcam import GradCAM as G
            from neuro.cortex_mapper import map_to_brain_region as M
            from neuro.brain_overlay import overlay_on_brain as O

            detect = d
            encode_image = e
            ActivationExtractor = A
            GradCAM = G
            map_to_brain_region = M
            overlay_on_brain = O


        # LAZY MODEL LOAD
        if extractor is None:

            print("Loading extractor...")
            extractor = ActivationExtractor()

        if gradcam is None:

            print("Loading gradcam...")
            gradcam = GradCAM()


        file = request.files["image"]

        if file.filename == "":
            return jsonify({"success": False, "error": "No file"})


        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)


        detections, _ = detect(path)

        embedding = encode_image(path)

        activations = extractor.run(path)

        heatmap_path = gradcam.generate(path)


        mapped = []

        for a in activations:

            mapped.append({
                "neuron": int(a["neuron"]),
                "activation": float(a["activation"]),
                "region": map_to_brain_region(a["neuron"])
            })


        brain_result_path = overlay_on_brain(mapped)


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

    app.run(host="0.0.0.0", port=port)
    