from ultralytics import YOLO

model = None


def get_model():

    global model

    if model is None:

        print("Loading YOLOv8n model...")

        # ✅ en hafif model
        model = YOLO("yolov8n.pt")

    return model


def detect(image_path):

    model = get_model()

    results = model(image_path)

    detections = []

    for r in results:

        for box in r.boxes:

            detections.append({

                "class": model.names[int(box.cls)],

                "confidence": float(box.conf),

                "x": float(box.xyxy[0][0]),
                "y": float(box.xyxy[0][1]),

            })

    return detections, results
