"""Reproduce every number and figure in the README: ``python run_analysis.py``."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from segmentation.cluster import compare_methods, fit_kmeans, profile_segments
from segmentation.data import FEATURES, load_customers, scale_features
from segmentation.kselect import evaluate_k, inertia_curve, suggest_k

OUT = Path(__file__).parent / "results"
SEED = 42
K = 6  # chosen after reading results/k_selection.csv; justification in the README


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = load_customers()
    scaled = scale_features(raw)
    X = scaled.to_numpy()

    scores = evaluate_k(X, seed=SEED)
    inertias = inertia_curve(X, seed=SEED)
    suggestions = suggest_k(scores, inertias)
    scores.round(4).to_csv(OUT / "k_selection.csv", index=False)
    print("k suggested by each criterion:", suggestions)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(range(1, len(inertias) + 1), inertias, "o-")
    axes[0].set_title("Elbow (inertia)")
    for ax, col in zip(axes[1:], ["silhouette", "davies_bouldin", "calinski_harabasz"]):
        ax.plot(scores["k"], scores[col], "o-")
        ax.set_title(col.replace("_", " ").title())
    for ax in axes:
        ax.axvline(K, color="red", ls="--", lw=1)
        ax.set_xlabel("k")
    fig.suptitle(f"Choosing the number of segments (red = chosen k={K})")
    fig.tight_layout()
    fig.savefig(OUT / "k_selection.png", dpi=150)
    plt.close(fig)

    labels = fit_kmeans(X, K, SEED)
    profile = profile_segments(raw, scaled, labels)
    profile.to_csv(OUT / "segment_profiles.csv")
    print(profile.to_string())

    comparison = compare_methods(X, K, SEED).round(3)
    comparison.to_csv(OUT / "method_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # Income vs spending is where the structure is clearest.
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg in range(K):
        m = labels == seg
        ax.scatter(raw.loc[m, FEATURES[1]], raw.loc[m, FEATURES[2]], s=35, label=f"Segment {seg}")
    ax.set_xlabel(FEATURES[1])
    ax.set_ylabel(FEATURES[2])
    ax.set_title(f"Customer segments (k-means, k={K})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "segments_income_vs_spending.png", dpi=150)
    plt.close(fig)

    z_cols = [c for c in profile.columns if c.endswith("(z)")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / K
    for i, (seg, row) in enumerate(profile[z_cols].iterrows()):
        ax.bar(np.arange(len(z_cols)) + i * width, row.values, width, label=f"Segment {seg}")
    ax.set_xticks(np.arange(len(z_cols)) + width * (K - 1) / 2)
    ax.set_xticklabels(FEATURES)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean z-score (0 = average customer)")
    ax.set_title("Segment profiles")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "segment_profiles.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
