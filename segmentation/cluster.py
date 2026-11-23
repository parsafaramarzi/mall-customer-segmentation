from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def fit_hierarchical(X: np.ndarray, k: int, linkage: str = "ward") -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)


def compare_methods(X: np.ndarray, k: int, seed: int = 42) -> pd.DataFrame:
    """Compare k-means with hierarchical linkages at the same k.

    Every method is scored with the same metric (Euclidean silhouette) on the
    same scaled data, so the numbers are directly comparable. ``ari_vs_kmeans``
    says how much each partition agrees with k-means (1.0 = identical).
    """
    km = fit_kmeans(X, k, seed)
    rows = [{"method": "kmeans", "silhouette": silhouette_score(X, km), "ari_vs_kmeans": 1.0}]
    for linkage in ("ward", "average", "complete", "single"):
        labels = fit_hierarchical(X, k, linkage)
        rows.append(
            {
                "method": f"hierarchical-{linkage}",
                "silhouette": silhouette_score(X, labels),
                "ari_vs_kmeans": adjusted_rand_score(km, labels),
            }
        )
    return pd.DataFrame(rows)


def profile_segments(raw: pd.DataFrame, scaled: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-segment size, raw feature means, z-score means and share of women."""
    z = scaled.assign(segment=labels).groupby("segment").mean().add_suffix(" (z)")
    out = raw.assign(segment=labels).groupby("segment").agg(
        size=("CustomerID", "size"),
        age=("Age", "mean"),
        income_k=("Annual Income (k$)", "mean"),
        spending=("Spending Score (1-100)", "mean"),
        pct_female=("Gender", lambda s: 100 * (s == "Female").mean()),
    )
    return out.join(z).round(2)
