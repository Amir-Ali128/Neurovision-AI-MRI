
import torch, cv2, numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

class GradCAM:

    def __init__(self):
        self.model = YOLO("yolov10n.pt")
        self.activations = None
        self.gradients = None
        self.register()

    def f_hook(self, m,i,o): self.activations=o.detach()
    def b_hook(self, m,gi,go): self.gradients=go[0].detach()

    def register(self):
        for n,l in self.model.model.named_modules():
            if isinstance(l, torch.nn.Conv2d):
                l.register_forward_hook(self.f_hook)
                l.register_backward_hook(self.b_hook)
                break

    def generate(self,path):
        img=Image.open(path).convert("RGB")
        t=transforms.Compose([transforms.Resize((640,640)),transforms.ToTensor()])
        x=t(img).unsqueeze(0)
        x.requires_grad=True

        out=self.model.model(x)
        loss=out[0].sum()
        self.model.model.zero_grad()
        loss.backward()

        grads=self.gradients[0]
        acts=self.activations[0]

        w=torch.mean(grads,dim=(1,2))
        cam=torch.zeros(acts.shape[1:],dtype=torch.float32)

        for i,val in enumerate(w):
            cam+=val*acts[i]

        cam=torch.relu(cam).cpu().numpy()
        cam=cv2.resize(cam,(640,640))
        cam-=cam.min()
        cam/=cam.max()
        return (cam*255).astype(np.uint8).tolist()
