# src/verifier.py

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class SSCDVerifier:
    def __init__(self, model_path: str):
        # Always use CPU for now
        self.device = "cpu"

        # Load TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # SSCD preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def get_embedding(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(img)

        emb = emb.cpu().numpy().flatten()
        emb = emb / np.linalg.norm(emb)  # L2 normalize

        return emb
