import torch
import cv2
import numpy as np
import clip
from PIL import Image

class GradCAM:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.model.eval()

    def generate(self, image_path):

        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        image.requires_grad = True

        # sadece image encoder kullan
        features = self.model.encode_image(image)

        loss = features.mean()

        self.model.zero_grad()

        loss.backward()

        grad = image.grad.cpu().numpy()[0]

        heatmap = np.mean(grad, axis=0)

        heatmap = np.maximum(heatmap, 0)

        heatmap /= heatmap.max() + 1e-8

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heatmap = cv2.resize(heatmap, (512,512))

        path = "static/heatmap.png"

        cv2.imwrite(path, heatmap)

        return path
