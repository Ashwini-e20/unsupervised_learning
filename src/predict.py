# src/predict.py

import os
import shutil
from PIL import Image

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import joblib

from .config import (
    IMAGE_SIZE,
    MODELS_DIR,
    DEVICE,
    CAT_RESULTS_DIR,
    DOG_RESULTS_DIR,
    UNKNOWN_RESULTS_DIR,
)

# >>> SET THIS based on your cluster inspection <<<
# Example: if you saw cluster 0 = cats, cluster 1 = dogs:
CLUSTER_TO_LABEL = {
    0: "cat",
    1: "dog",
    # flip if needed:
    # 0: "dog",
    # 1: "cat",
}


def get_resnet_feature_extractor():
    """Load pretrained ResNet50 and remove last FC layer."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1]  # remove final classification layer
    model = nn.Sequential(*modules)
    model.to(DEVICE)
    model.eval()
    return model


def load_pca_kmeans():
    """Load PCA and KMeans models from disk."""
    pca_path = os.path.join(MODELS_DIR, "pca.pkl")
    kmeans_path = os.path.join(MODELS_DIR, "kmeans.pkl")

    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        pca = None

    kmeans = joblib.load(kmeans_path)
    return pca, kmeans


def preprocess_image(img_path: str) -> torch.Tensor:
    """Load and preprocess an image into a model-ready tensor."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # [1, C, H, W]
    return img


def predict_image(img_path: str):
    """
    Predict cluster + label ("cat"/"dog") for a single image.
    Returns (label, cluster_id).
    """
    device = DEVICE
    model = get_resnet_feature_extractor()
    pca, kmeans = load_pca_kmeans()

    img_tensor = preprocess_image(img_path).to(device)

    with torch.no_grad():
        feats = model(img_tensor)          # [1, 2048, 1, 1]
        feats = feats.view(1, -1).cpu().numpy()  # [1, 2048]

    if pca is not None:
        feats = pca.transform(feats)

    cluster_id = kmeans.predict(feats)[0]
    label = CLUSTER_TO_LABEL.get(cluster_id, f"unknown_cluster_{cluster_id}")
    return label, cluster_id


def save_image_to_result_folder(img_path: str, label: str) -> str:
    """
    Copy image into results/cat or results/dog (or unknown).
    Returns final destination path.
    """
    # Choose destination dir based on label
    if label.lower() == "cat":
        dest_dir = CAT_RESULTS_DIR
    elif label.lower() == "dog":
        dest_dir = DOG_RESULTS_DIR
    else:
        dest_dir = UNKNOWN_RESULTS_DIR

    os.makedirs(dest_dir, exist_ok=True)

    # Keep same filename, just in new folder
    filename = os.path.basename(img_path)
    dest_path = os.path.join(dest_dir, filename)

    # If file with same name already exists, avoid overwrite by adding suffix
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        i = 1
        while True:
            new_name = f"{name}_{i}{ext}"
            new_dest = os.path.join(dest_dir, new_name)
            if not os.path.exists(new_dest):
                dest_path = new_dest
                break
            i += 1

    shutil.copy2(img_path, dest_path)
    return dest_path


def classify_and_store(img_path: str):
    """
    Full pipeline:
      1. Predict cat/dog
      2. Store image in corresponding results subfolder
      3. Return (label, cluster_id, stored_path)
    """
    label, cluster_id = predict_image(img_path)
    stored_path = save_image_to_result_folder(img_path, label)
    return label, cluster_id, stored_path


if __name__ == "__main__":
    # Run from command line:
    # python -m src.predict path/to/test_image.jpg
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict cat/dog for an image and store it in results folder."
    )
    parser.add_argument("img_path", type=str, help="Path to input image")

    args = parser.parse_args()
    img_path = args.img_path

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    label, cid, stored = classify_and_store(img_path)

    print(f"Input image : {img_path}")
    print(f"Cluster ID  : {cid}")
    print(f"Predicted   : {label}")
    print(f"Stored at   : {stored}")
