
from ultralytics import YOLO

model = YOLO("yolov10n.pt")

def detect(image_path):
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf)
            })
    return detections, results
