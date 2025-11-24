"""KMeans clustering model for food items.

This module:
- Loads and cleans the food dataset using the data preparation utilities.
- Scales the nutritional features.
- Tries several values of k and computes clustering quality metrics.
- Selects the best model based on the silhouette score.
- Adds the chosen cluster labels to the cleaned dataframe.
- Optionally saves the clustered dataframe and the trained model to disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Import your existing data preparation functions
from nutrimap_app.data_prep_marie import clean_food_data, scale_food_data


# Default paths (adjust if your project structure is different)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data/processed"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
CLUSTERED_DATA_PATH = DATA_DIR / "food_with_clusters.csv"


def _prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the cleaning and scaling steps and return both dataframes.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned (unscaled) dataframe, including the `food_item` column.
    scaled_df : pd.DataFrame
        Scaled dataframe, with the same row order as df_clean.
    """

    df_clean = clean_food_data()
    scaled_df = scale_food_data()
    return df_clean, scaled_df


def build_kmeans_model(
    random_state: int = 42,
    save_model: bool = True,
    save_data: bool = True,
):
    """Build a fixed 3‑cluster KMeans model.

    Returns
    -------
    best_model : KMeans
        The fitted KMeans instance.
    df_with_clusters : pd.DataFrame
        Cleaned dataframe with an added `cluster` column.
    """

    df_clean, scaled_df = _prepare_data()

    # Features: all scaled numeric columns except the identifier
    X = scaled_df.drop(columns=["food_item"], errors="ignore")

    # Fixed k=3
    k = 3
    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(X)

    # Attach clusters
    df_with_clusters = df_clean.copy()
    df_with_clusters["cluster"] = labels

    # Save outputs
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with BEST_MODEL_PATH.open("wb") as f:
            pickle.dump(model, f)

    if save_data:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_with_clusters.to_csv(CLUSTERED_DATA_PATH, index=False)

    return model, df_with_clusters


def kmeanModel(
    random_state: int = 42,
    save_model: bool = True,
    save_data: bool = True,
):
    return build_kmeans_model(
        random_state=random_state,
        save_model=save_model,
        save_data=save_data,
    )


if __name__ == "__main__":
    model, df_clusters = build_kmeans_model()
    print("KMeans clustering completed with k=3.")
    print("Model saved to:", BEST_MODEL_PATH)
    print("Clustered data saved to:", CLUSTERED_DATA_PATH)
