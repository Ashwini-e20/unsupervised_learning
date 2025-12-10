# src/inspect_clusters.py

import pandas as pd
import os
from .config import MODELS_DIR

def show_samples_per_cluster(n_samples=5):
    csv_path = os.path.join(MODELS_DIR, "cluster_assignments.csv")
    df = pd.read_csv(csv_path)

    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\nCluster {cluster_id}:")
        sample_paths = df[df["cluster"] == cluster_id]["path"].head(n_samples)
        for p in sample_paths:
            print("  ", p)
        print("  ---> Open these images manually and see if they look like CAT or DOG.")

if __name__ == "__main__":
    show_samples_per_cluster()
