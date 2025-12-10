# main.py

from src.feature_extractor import extract_and_save_features
from src.cluster import cluster_features
from src.inspect_clusters import show_samples_per_cluster

if __name__ == "__main__":
    # 1. Extract features with ResNet and save to models/features.npz
    extract_and_save_features()

    # 2. Cluster features with PCA + KMeans
    cluster_features()

    # 3. Show some sample paths from each cluster so YOU can decide mapping
    show_samples_per_cluster()

    print("\nNow open those sample image paths on disk and decide:")
    print("Which cluster ID is 'cat' and which is 'dog'?")
    print("Then update CLUSTER_TO_LABEL in src/predict.py accordingly.")
