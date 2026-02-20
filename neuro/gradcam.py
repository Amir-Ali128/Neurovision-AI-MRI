import torch
import cv2
import numpy as np
from ultralytics import YOLO

class GradCAM:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.model.eval()

        # gradient açık olacak
        for param in self.model.model.parameters():
            param.requires_grad = True

        self.activations = None
        self.gradients = None

        target_layer = self.model.model.model[-2]

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_path):

        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (640,640))
        img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0)
        img_tensor.requires_grad = True

        results = self.model.model(img_tensor)

        score = results[0].mean()

        self.model.model.zero_grad()
        score.backward()

        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        weights = np.mean(gradients, axis=(1,2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam = cam - cam.min()
        cam = cam / cam.max()

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        overlay = heatmap * 0.4 + img * 0.6

        output_path = image_path.replace(".jpg", "_gradcam.jpg")
        cv2.imwrite(output_path, overlay)

        return output_path
