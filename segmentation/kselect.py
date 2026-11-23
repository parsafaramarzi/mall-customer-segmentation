from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def elbow_k(inertias: np.ndarray) -> int:
    """Point of an inertia curve farthest from the straight line joining its ends.

    ``inertias[i]`` is the inertia for k = i + 1.
    """
    x = np.arange(1, len(inertias) + 1, dtype=float)
    y = np.asarray(inertias, dtype=float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    dist = np.abs(dx * (y[0] - y) - dy * (x[0] - x)) / np.hypot(dx, dy)
    return int(np.argmax(dist) + 1)


def evaluate_k(X: np.ndarray, k_max: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Score k-means for k = 2..k_max with three internal validity indices.

    ``k_max`` defaults to sqrt(n): Davies-Bouldin keeps improving at very large k,
    so an uncapped sweep would just reward over-segmentation.
    """
    k_max = k_max or int(np.sqrt(len(X)))
    rows = []
    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def inertia_curve(X: np.ndarray, k_max: int = 30, seed: int = 42) -> np.ndarray:
    return np.array(
        [KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).inertia_ for k in range(1, k_max + 1)]
    )


def suggest_k(scores: pd.DataFrame, inertias: np.ndarray) -> dict[str, int]:
    """Best k per criterion. The criteria disagree on purpose; see the README."""
    return {
        "elbow": elbow_k(inertias),
        "silhouette": int(scores.loc[scores["silhouette"].idxmax(), "k"]),
        "davies_bouldin": int(scores.loc[scores["davies_bouldin"].idxmin(), "k"]),
        "calinski_harabasz": int(scores.loc[scores["calinski_harabasz"].idxmax(), "k"]),
    }
