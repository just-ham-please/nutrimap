import pandas as pd
import random
from pathlib import Path

###
# Optimzation function: currently this is just a dummy, working with the dummy file.
# We'll need to change the path, once we have the correct file.
# We'll also need to add the logic here.
###

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "test_data_justus" / "foods_ui_dummy.csv"

df_ui = pd.read_csv(CSV_PATH)

def optimize_plate(protein: str, carb: str, fat: str):
    protein_list = df_ui[df_ui["plate_role"] == "protein"]["food_name"].tolist()
    carb_list = df_ui[df_ui["plate_role"] == "carb"]["food_name"].tolist()
    fat_list = df_ui[df_ui["plate_role"] == "fat"]["food_name"].tolist()

    # Dummy: random pick (optional: nicht dieselbe wie Auswahl)
    better_protein = random.choice([x for x in protein_list if x != protein] or protein_list)
    better_carb = random.choice([x for x in carb_list if x != carb] or carb_list)
    better_fat = random.choice([x for x in fat_list if x != fat] or fat_list)

    return {
        "better_protein": better_protein,
        "better_carb": better_carb,
        "better_fat": better_fat
    }
