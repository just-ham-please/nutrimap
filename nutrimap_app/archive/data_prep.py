import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------
# CONSTANT: Columns to keep in the cleaned dataset
# -------------------------------------------------------
COLUMNS_TO_KEEP = [
    "food",
    "Caloric Value",
    "Fat",
    "Saturated Fats",
    "Carbohydrates",
    "Sugars",
    "Protein",
    "Dietary Fiber"
]
RENAME_COLUMNS = {
    "Caloric Value": "energy_kcal",
    "Fat": "fat_g",
    "Saturated Fats": "satfat_g",
    "Carbohydrates": "carbs_g",
    "Sugars": "sugars_g",
    "Protein": "protein_g",
    "Dietary Fiber": "fiber_g"
}

# -------------------------------------------------------
# Default paths
# -------------------------------------------------------
RAW_FOLDER = "../raw_data"
CLEANED_PATH = "../data/processed/foods_cleaned.csv"
SCALED_PATH = "../data/processed/foods_scaled.csv"

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
    # Load all CSV files and merge into df
    # ---------------------------------------------------
    files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    if not files:
        raise ValueError(f"No CSV files found in folder: {raw_folder}")

    df_list = []
    for f in files:
        file_path = raw_folder / f
        df_temp = pd.read_csv(file_path)
        df_temp["source_file"] = f
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    # ---------------------------------------------------
    # select columns and rename
    # ---------------------------------------------------

    df_clean = df[COLUMNS_TO_KEEP].copy()

    df_clean = df_clean.rename(columns=RENAME_COLUMNS)

    # ---------------------------------------------------
    # Drop duplicates and NaN Checke
    # ---------------------------------------------------
    df_clean = df_clean.drop_duplicates()
    print("NaNs per column:\n", df_clean.isna().sum())

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
    feature_cols = [col for col in numeric_cols if col != "food"]

    df_scaled = df_clean.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

    output_scaled_path = Path(output_scaled_path)
    output_scaled_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_scaled_path, index=False)

    return df_scaled
