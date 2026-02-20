import torch
import clip
from PIL import Image

device = "cpu"

# Lazy load variables
model = None
preprocess = None


def load_model():

    global model, preprocess

    if model is None:

        print("Loading CLIP model...")

        model, preprocess = clip.load("ViT-B/32", device=device)

        model.eval()

    return model, preprocess


def encode_image(path):

    model, preprocess = load_model()

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    with torch.no_grad():

        embedding = model.encode_image(image)

    return embedding.cpu().numpy().tolist()
