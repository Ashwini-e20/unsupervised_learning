# src/config.py

import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "images_unlabeled")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# Clustering / PCA
USE_PCA = True
PCA_COMPONENTS = 100
N_CLUSTERS = 2

# Device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === NEW: results directories for classified images ===
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CAT_RESULTS_DIR = os.path.join(RESULTS_DIR, "cat")
DOG_RESULTS_DIR = os.path.join(RESULTS_DIR, "dog")
UNKNOWN_RESULTS_DIR = os.path.join(RESULTS_DIR, "unknown")

for d in (RESULTS_DIR, CAT_RESULTS_DIR, DOG_RESULTS_DIR, UNKNOWN_RESULTS_DIR):
    os.makedirs(d, exist_ok=True)