from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "Mall_Customers.csv"


def load_customers(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Load the raw customer table (CustomerID, Gender, Age, Income, Spending)."""
    df = pd.read_csv(path)
    missing = {"CustomerID", "Gender", *FEATURES} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score the clustering features so no single feature dominates distances.

    Gender is deliberately excluded: it is kept aside and only used afterwards
    to check whether the segments happen to be gender-skewed.
    """
    scaled = StandardScaler().fit_transform(df[FEATURES])
    return pd.DataFrame(scaled, columns=FEATURES, index=df.index)
