
import torch
from ultralytics import YOLO

class ActivationExtractor:

    def __init__(self):
        self.model = YOLO("yolov10n.pt")
        self.activations = []
        self.register()

    def hook(self, module, input, output):
        flat = output.detach().cpu().numpy().flatten()
        self.activations = [
            {"neuron": i, "activation": float(v)}
            for i, v in enumerate(flat[:800])
        ]

    def register(self):
        for name, layer in self.model.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(self.hook)
                break

    def run(self, path):
        self.model(path)
        return self.activations
