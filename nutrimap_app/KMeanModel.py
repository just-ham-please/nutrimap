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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Import your existing data preparation functions
from nutrimap_app.data_prep_final import clean_food_data, scale_food_data


# Default paths (adjust if your project structure is different)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data/processed"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
CLUSTERED_DATA_PATH = DATA_DIR / "food_with_clusters.csv"
SUBCLUSTERING_DATA_PATH = DATA_DIR / "food_with_subclusters.csv"
PLATE_ROLE_DATA_PATH = DATA_DIR / "food_with_plate_roles.csv"


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
    """Build a fixed 5-cluster KMeans model.

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

    # Fixed k=5
    k = 5
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

def subclustering(df_with_clusters, save_data: bool = True):
    ''' This function takes in a dataframe with clusters,
    runs subclustering and
    returns a dataframe with subclusters added.
    '''
    df = df_with_clusters.copy()

    # df with rest of clusters - 2,3,4
    df_rest = df[df["cluster"] > 1].copy()

    # Run subclustering
    zero_cluster_df = df[df["cluster"] == 0].copy()
    one_cluster_df = df[df["cluster"] == 1].copy()
    X_zero = zero_cluster_df.drop(columns=["food_item", "cluster"])
    X_one = one_cluster_df.drop(columns=["food_item", "cluster"])

    scaler = MinMaxScaler()

    X_zero = scaler.fit_transform(X_zero)
    X_one = scaler.fit_transform(X_one)

    # Subcluster 0
    model0 = KMeans(n_clusters=2, random_state=42)
    labels0 = model0.fit_predict(X_zero)
    zero_cluster_df["subcluster"] = labels0

    # Subcluster 1
    model1 = KMeans(n_clusters=2, random_state=42)
    labels1 = model1.fit_predict(X_one)
    one_cluster_df["subcluster"] = labels1

    df = pd.concat([zero_cluster_df, one_cluster_df,df_rest], axis=0).fillna(0)
    df["subcluster"] = df["subcluster"].astype(int)

    # Create the supercluster column by joining the cluster and subcluster columns
    df["supercluster"] = df["cluster"].astype(str) + "-" + df["subcluster"].astype(str)

    # Save outputs
    if save_data:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SUBCLUSTERING_DATA_PATH, index=False)

    return df

def assign_plate_role( df_with_subclusters: pd.DataFrame, save_data: bool = True) -> pd.DataFrame:
    """
    Add a 'plate_role' column based on the following mapping:
        cluster = 0 and subcluster = 0 → 'fruit / veg'
        cluster = 0 and subcluster = 1 → 'protein'
        cluster = 1 and subcluster = 0 → 'carbs'
        cluster = 2                    → 'protein'
        cluster = 3                    → 'fat'
        cluster = 4                    → 'protein'
        all others                     → 'other'
    Parameters
    ----------
    df_with_subclusters : pd.DataFrame
        Dataframe returned by subclustering(), must contain
        'cluster' and 'subcluster' columns.
    Returns
    -------
    pd.DataFrame
        Updated dataframe including a new 'plate_role' column.
    """
    df = df_with_subclusters.copy()
    # Default role for anything not mapped above
    df["plate_role"] = "other"
    # Cluster 0
    df.loc[(df["cluster"] == 0) & (df["subcluster"] == 0), "plate_role"] = "fruit / veg"
    df.loc[(df["cluster"] == 0) & (df["subcluster"] == 1), "plate_role"] = "protein"
    # Cluster 1
    df.loc[(df["cluster"] == 1) & (df["subcluster"] == 0), "plate_role"] = "carbs"
    # Cluster 2
    df.loc[df["cluster"] == 2, "plate_role"] = "protein"
    # Cluster 3
    df.loc[df["cluster"] == 3, "plate_role"] = "fat"
    # Cluster 4
    df.loc[df["cluster"] == 4, "plate_role"] = "protein"

        # Save outputs
    if save_data:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PLATE_ROLE_DATA_PATH, index=False)

    return df


if __name__ == "__main__":
    model, df_clusters = build_kmeans_model()
    print("KMeans clustering completed with k=5.")
    print("Model saved to:", BEST_MODEL_PATH)
    print("Clustered data saved to:", CLUSTERED_DATA_PATH)
