import torch
from ultralytics import YOLO


class ActivationExtractor:

    def __init__(self):

        self.model = YOLO("yolov10n.pt")

        self.activations = []

        self.register()


    def hook(self, module, input, output):

        flat = output.detach().cpu().numpy().flatten()

        self.activations = []

        for i, v in enumerate(flat[:300]):

            self.activations.append({

                "neuron": int(i),

                "activation": float(v)

            })


    def register(self):

        for name, layer in self.model.model.named_modules():

            if isinstance(layer, torch.nn.Conv2d):

                layer.register_forward_hook(self.hook)

                break


    def run(self, path):

        self.activations = []

        self.model(path)

        if self.activations is None:
            return []

        return self.activations
