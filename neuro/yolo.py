
from ultralytics import YOLO

model = None

def get_model():
    global model
    if model is None:
        model = YOLO("yolov10n.pt")
    return model

def detect(image_path):
    model = get_model()

    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf)
            })

    return detections, results
