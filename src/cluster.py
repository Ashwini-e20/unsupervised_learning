# src/cluster.py

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from .config import MODELS_DIR, USE_PCA, PCA_COMPONENTS, N_CLUSTERS


def cluster_features():
    data = np.load(os.path.join(MODELS_DIR, "features.npz"), allow_pickle=True)
    X = data["features"]
    paths = data["paths"]

    print(f"Loaded features: {X.shape}")

    # Optional PCA
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        X_reduced = pca.fit_transform(X)
        joblib.dump(pca, os.path.join(MODELS_DIR, "pca.pkl"))
        print(f"PCA reduced from {X.shape[1]} → {PCA_COMPONENTS} dimensions")
    else:
        pca = None
        X_reduced = X

    # KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)
    # 1) Print how many images per cluster
    unique, counts = np.unique(clusters, return_counts=True)
    print("Cluster distribution:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} images") 
    # 2) Print KMeans inertia (within-cluster sum of squares)
    print(f"KMeans inertia: {kmeans.inertia_:.2f}") 
    
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    print(f"KMeans fitted with {N_CLUSTERS} clusters")

    # Evaluate with silhouette score (just a sanity check)
    if X_reduced.shape[0] >= N_CLUSTERS + 1:
        score = silhouette_score(X_reduced, clusters)
        print(f"Silhouette score: {score:.4f}")

    # Save cluster assignments for inspection
    out_csv = os.path.join(MODELS_DIR, "cluster_assignments.csv")
    import pandas as pd
    df = pd.DataFrame({"path": paths, "cluster": clusters})
    df.to_csv(out_csv, index=False)
    print(f"Saved cluster assignments to {out_csv}")


if __name__ == "__main__":
    cluster_features()
