# src/feature_extractor.py

import numpy as np
import torch
from torch import nn
from torchvision import models
from tqdm import tqdm
import os

from .data_loader import create_dataloader
from .config import DEVICE, MODELS_DIR


def get_resnet_feature_extractor():
    # Pretrained ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Remove the final classification layer (fc) → use features from penultimate layer
    modules = list(resnet.children())[:-1]  # remove last FC
    model = nn.Sequential(*modules)
    model.to(DEVICE)
    model.eval()
    return model


def extract_and_save_features():
    dataloader = create_dataloader()
    model = get_resnet_feature_extractor()

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            feats = model(images)  # shape: [batch, 2048, 1, 1]
            feats = feats.view(feats.size(0), -1)  # [batch, 2048]

            all_features.append(feats.cpu().numpy())
            all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)

    out_path = os.path.join(MODELS_DIR, "features.npz")
    np.savez(out_path, features=all_features, paths=np.array(all_paths))
    print(f"Saved features to {out_path}")


if __name__ == "__main__":
    extract_and_save_features()
