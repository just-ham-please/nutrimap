import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# -------------------------------------------------------
# CONSTANT: Columns to keep in the cleaned dataset
# -------------------------------------------------------
COLUMNS_TO_KEEP = [
    "food_item",
    "Energy",
    "data_type",
    "Total lipid (fat)",
    "Fatty acids, total saturated",
    "Carbohydrate, by difference",
    "Protein",
    "Fiber, total dietary"
]
RENAME_COLUMNS = {
    "Energy": "energy_kcal",
    "Total lipid (fat)": "fat_g",
    "Fatty acids, total saturated": "satfat_g",
    "Carbohydrate, by difference": "carbs_g",
    "Protein": "protein_g",
    "Fiber, total dietary": "fiber_g"
}

# -------------------------------------------------------
# Default paths
# -------------------------------------------------------
RAW_FOLDER = "raw_data"
CLEANED_PATH = "data/processed/foods_cleaned_marie.csv"
SCALED_PATH = "data/processed/foods_scaled_marie.csv"

# -------------------------------------------------------
# Function for data cleaning
# -------------------------------------------------------

def clean_food_data(raw_folder: str = RAW_FOLDER,
                    output_path: str = CLEANED_PATH):
    """
    End-to-end function cleaning nutrition data.

    Steps performed:
    Load all CSV files from the raw_folder.
    Merge them into one combined DataFrame.
    Select only the relevant nutrient columns.
    ename the selected columns for consistency.
    Remove duplicate rows.
    Save the cleaned dataset to output_path.

    Parameters
    ----------
    raw_folder : str or Path
        Directory containing the raw CSV files.
    output_path : str or Path
        File path where the processed dataset will be saved.

    Returns
    -------
    pandas.DataFrame
       cleaned dataframe
    """

    raw_folder = Path(raw_folder)

    # ---------------------------------------------------
    # Load parquet file
    # ---------------------------------------------------
    file_path = "../raw_data/data.parquet"
    df = pd.read_parquet(file_path, engine="fastparquet")


    # ---------------------------------------------------
    # select columns and rename
    # ---------------------------------------------------

    df_clean = df[COLUMNS_TO_KEEP].copy()

    df_clean = df_clean.rename(columns=RENAME_COLUMNS)



    # ---------------------------------------------------
    # remove branded foods from the dataset 500k ->7k
    # ---------------------------------------------------
    df_clean = df_clean[df_clean["data_type"] != "branded_food"].reset_index(drop=True)


    # ---------------------------------------------------
    # add column with energy in kcal calculated
    # ---------------------------------------------------
    df_clean["energy_kcal_calculated"] = (
        df_clean["fat_g"] * 9 +
        df_clean["carbs_g"] * 4 +
        df_clean["protein_g"] * 4
    ).round(1)

    # ---------------------------------------------------
    # replace NaN in sat fat column with 0 where likely 0
    # ---------------------------------------------------

    df_clean["satfat_g"] = np.where(
    (df_clean["fat_g"] < 3) &
    (df_clean["satfat_g"].isna()),
    0,
    df_clean["satfat_g"]
    )

    # ---------------------------------------------------
    # replace NaN in fiber column with 0 where likely 0
    # ---------------------------------------------------

    df_clean["fiber_g"] = np.where(
    (df_clean["carbs_g"] < 5) &
    (df_clean["fiber_g"].isna()),
    0,
    df_clean["fiber_g"]
    )



    # ---------------------------------------------------
    # Drop duplicates and drop NaN
    # ---------------------------------------------------
    df_clean = df_clean.drop_duplicates()

    """the cols_required are mandatory for modelling, in case they are NaN they are dropped
    CAVE: there will reamain NaNs in fibre and saturated fats!"""

    cols_required = [
    "food_item",
    "energy_kcal_calculated",
    "fat_g",
    "carbs_g",
    "protein_g",
    "fiber_g",
    "satfat_g"
    ]

    df_clean = df_clean.dropna(subset=cols_required)


    # ---------------------------------------------------
    # Remove column data_type and energy_kcal
    # ---------------------------------------------------
    df_clean = df_clean.drop(columns=["data_type", "energy_kcal"])


    # ---------------------------------------------------
    # Save output
    # ---------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    return df_clean




# -------------------------------------------------------
# Function for data scaling
# -------------------------------------------------------

def scale_food_data(input_clean_path: str = CLEANED_PATH,
                    output_scaled_path: str = SCALED_PATH):

    """
    Scale numeric nutrient features in the cleaned dataset using MinMaxScaler (0–1).

    Steps performed:
    Load the cleaned dataset from input_clean_path.
    Identify all numeric features except the “food” column.
    Apply MinMax scaling to normalize numeric values to the range 0–1.
    Save the scaled dataset to output_scaled_path.

    Parameters
    ----------
    input_clean_path : str or Path
        File path to the cleaned dataset created by clean_food_data().
        This is the dataset that will be scaled.

    output_scaled_path : str or Path
        File path where the scaled dataset will be saved.

    Returns
    -------
    pandas.DataFrame
        A scaled version of the cleaned dataset.
    """

    df_clean = pd.read_csv(input_clean_path)

    scaler = MinMaxScaler()

    numeric_cols = df_clean.select_dtypes(include=["float64", "int64"]).columns
    feature_cols = [col for col in numeric_cols if col != "food_item"]

    df_scaled = df_clean.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

    output_scaled_path = Path(output_scaled_path)
    output_scaled_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_scaled_path, index=False)

    return df_scaled
