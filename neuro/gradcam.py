import torch
import cv2
import numpy as np

class GradCAM:

    def __init__(self):

        from neuro.clip_model import model

        self.model = model

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

    def generate(self, image_path):

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224,224))

        tensor = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)

        tensor.requires_grad = True

        output = self.model(tensor)

        loss = output.mean()

        self.model.zero_grad()

        loss.backward()

        grad = tensor.grad[0].permute(1,2,0).detach().numpy()

        heatmap = np.mean(grad, axis=2)

        heatmap = np.maximum(heatmap,0)

        heatmap /= heatmap.max() + 1e-8

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        path = "static/heatmap.png"

        cv2.imwrite(path, heatmap)

        return path
