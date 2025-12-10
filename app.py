# app.py

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.metrics import silhouette_score
import joblib

from src.predict import classify_and_store
from src.config import MODELS_DIR, RESULTS_DIR

# ------------- Helpers for metrics ------------- #

@st.cache_data
def load_clustering_metrics():
    """
    Load PCA + KMeans + features and compute:
      - silhouette score
      - cluster distribution
    Uses the already saved models from training.
    """
    features_path = os.path.join(MODELS_DIR, "features.npz")
    kmeans_path = os.path.join(MODELS_DIR, "kmeans.pkl")
    pca_path = os.path.join(MODELS_DIR, "pca.pkl")
    cluster_csv = os.path.join(MODELS_DIR, "cluster_assignments.csv")

    if not (os.path.exists(features_path) and os.path.exists(kmeans_path)):
        return None  # training not run yet

    data = np.load(features_path, allow_pickle=True)
    X = data["features"]

    # Load PCA if it exists
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        X_reduced = pca.transform(X)
    else:
        pca = None
        X_reduced = X

    # Load KMeans
    kmeans = joblib.load(kmeans_path)

    # Use existing cluster assignments if CSV exists,
    # otherwise compute from KMeans
    if os.path.exists(cluster_csv):
        df = pd.read_csv(cluster_csv)
        clusters = df["cluster"].values
    else:
        clusters = kmeans.predict(X_reduced)
        df = pd.DataFrame({"path": data["paths"], "cluster": clusters})

    # Silhouette score
    if len(np.unique(clusters)) > 1:
        sil_score = float(silhouette_score(X_reduced, clusters))
    else:
        sil_score = None

    # Cluster distribution
    unique, counts = np.unique(clusters, return_counts=True)
    dist_df = pd.DataFrame({"cluster": unique, "count": counts})

    return {
        "silhouette": sil_score,
        "distribution": dist_df,
        "assignments": df,
    }


def save_uploaded_file_to_temp(uploaded_file):
    """
    Save a Streamlit UploadedFile to a temp file on disk
    so we can pass a real path to classify_and_store().
    """
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    return temp_path


# ------------- Streamlit UI ------------- #

st.set_page_config(page_title="Cat vs Dog (Unsupervised)")

st.title("Unsupervised Cat vs Dog Classifier")
st.write(
    """
This app uses **unsupervised learning** (pretrained CNN embeddings + KMeans)
to cluster images into two groups and then interpret them as **cats** vs **dogs**.
"""
)

st.sidebar.header("Model Metrics")

metrics = load_clustering_metrics()
if metrics is None:
    st.sidebar.warning(
        "Metrics not available. Run `python main.py` first to train and cluster."
    )
else:
    sil = metrics["silhouette"]
    dist_df = metrics["distribution"]

    if sil is not None:
        st.sidebar.metric("Silhouette score", f"{sil:.4f}")
    else:
        st.sidebar.write("Silhouette score: not available (need at least 2 clusters).")

    st.sidebar.subheader("Cluster distribution")
    st.sidebar.dataframe(dist_df)

    st.sidebar.caption(
        "Higher silhouette ≈ better separation. Distribution shows how many images per cluster."
    )

st.write("---")

st.header(" Classify an image")

uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"],
    help="The model will predict cat/dog and store the image in the corresponding results subfolder.",
)

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            # Save to a temp file, then classify
            temp_path = save_uploaded_file_to_temp(uploaded_file)
            label, cluster_id, stored_path = classify_and_store(temp_path)

        st.success(f"Prediction: **{label.upper()}** (cluster {cluster_id})")
        st.write(f"Image stored at: `{stored_path}`")

        # Small note about where results live
        st.info(f"All classified images are available under the `results/` folder.\n\nBase results dir: `{RESULTS_DIR}`")

st.write("---")
st.caption(
    "Tip: make sure you've run `python main.py` at least once so the features and clustering are trained."
)
