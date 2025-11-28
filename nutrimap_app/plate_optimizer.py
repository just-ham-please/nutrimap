import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. GLOBAL SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
# Data from the food with clusters file
CSV_PATH = BASE_DIR / './data/processed/food_with_plate_roles.csv'

# Nutrients we sum up and compare
NUTRIENT_COLS = [
    "energy_kcal",
    "fat_g",
    # "satfat_g",
    "carbs_g",
    "protein_g",
    "fiber_g"
]

# Optimal plate reference values (simple example)
OPTIMAL_PLATE = {
    "energy_kcal": 666,
    "protein_g": 17,
    "carbs_g": 86,
    "fat_g": 23,
    "fiber_g": 10
}

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_food_data():
    """Loads the dummy CSV into a DF with food_item as index."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Food CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Safety: ensure required columns exist
    required_cols = ["food_item", "plate_role"] + NUTRIENT_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.set_index("food_item")
    return df


# Load once at startup
FOODS_DF = load_food_data()


# ------------------------------------------------------------
# 3. COMPUTE NUTRIENTS FOR GIVEN PLATE
# ------------------------------------------------------------

def compute_plate_nutrients(ingredients):
    """
    ingredients: list of dicts:
    [
      {"role": "protein", "food_name": "Chicken Breast", "grams": 150},
      ...
    ]

    Returns dict of summed nutrients.
    """
    totals = {n: 0.0 for n in NUTRIENT_COLS}

    for ing in ingredients:
        food = ing["food_name"]
        grams = float(ing["grams"])
        if grams < 0:
            grams = 0

        if food not in FOODS_DF.index:
            raise ValueError(f"Food item not found: {food}")

        row = FOODS_DF.loc[food]
        factor = grams / 100.0

        for n in NUTRIENT_COLS:
            totals[n] += row[n] * factor

    return totals


# ------------------------------------------------------------
# 4. COMPARE WITH OPTIMAL PLATE (GAP ANALYSIS)
# ------------------------------------------------------------

def compare_with_optimal(actual):
    """
    actual: dict of nutrient -> actual value

    Returns dict of nutrient -> {actual, target, delta, delta_pct}
    """
    results = {}

    for nutrient in NUTRIENT_COLS:
        target = OPTIMAL_PLATE.get(nutrient, 0)
        a = actual.get(nutrient, 0)
        delta = a - target
        delta_pct = (delta / target * 100) if target > 0 else None

        results[nutrient] = {
            "actual": round(a, 2),
            "target": target,
            "delta": round(delta, 2),
            "delta_pct": round(delta_pct, 2) if delta_pct is not None else None
        }

    return results


# ------------------------------------------------------------
# 5. MAIN EXPORTED FUNCTION FOR API
# ------------------------------------------------------------

def analyze_plate(ingredients):
    """
    Main entrypoint used by the API.
    Takes list of:
    {
      "role": "protein",
      "food_name": "...",
      "grams": 120
    }

    Returns:
    {
      "actual": {...},
      "target": OPTIMAL_PLATE,
      "gaps": {...}
    }
    """
    actual = compute_plate_nutrients(ingredients)
    gaps = compare_with_optimal(actual)

    return {
        "actual": actual,
        "target": OPTIMAL_PLATE,
        "gaps": gaps
    }
