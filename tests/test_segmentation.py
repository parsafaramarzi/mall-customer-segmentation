import numpy as np
import pytest

from segmentation.cluster import compare_methods, fit_kmeans, profile_segments
from segmentation.data import FEATURES, load_customers, scale_features
from segmentation.kselect import elbow_k, evaluate_k


@pytest.fixture(scope="module")
def raw():
    return load_customers()


def test_dataset_shape_and_columns(raw):
    assert len(raw) == 200
    assert set(FEATURES) <= set(raw.columns)
    assert raw[FEATURES].notna().all().all()


def test_scaling_gives_zero_mean_unit_variance(raw):
    scaled = scale_features(raw)
    assert np.allclose(scaled.mean(), 0, atol=1e-9)
    assert np.allclose(scaled.std(ddof=0), 1, atol=1e-9)


def test_elbow_finds_obvious_bend():
    inertias = np.array([1000, 400, 150, 140, 135, 132, 130, 129])
    assert elbow_k(inertias) == 3


def test_kmeans_is_deterministic_for_a_seed(raw):
    X = scale_features(raw).to_numpy()
    assert np.array_equal(fit_kmeans(X, 5, seed=1), fit_kmeans(X, 5, seed=1))


def test_k_scores_cover_requested_range(raw):
    X = scale_features(raw).to_numpy()
    scores = evaluate_k(X, k_max=6)
    assert list(scores["k"]) == [2, 3, 4, 5, 6]
    assert scores["silhouette"].between(-1, 1).all()


def test_profiles_account_for_every_customer(raw):
    scaled = scale_features(raw)
    labels = fit_kmeans(scaled.to_numpy(), 6)
    profile = profile_segments(raw, scaled, labels)
    assert profile["size"].sum() == len(raw)


def test_comparison_reports_kmeans_agreement_of_one_with_itself(raw):
    X = scale_features(raw).to_numpy()
    table = compare_methods(X, 6)
    assert table.loc[table["method"] == "kmeans", "ari_vs_kmeans"].iloc[0] == 1.0
