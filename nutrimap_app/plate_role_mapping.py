import numpy as np
import pandas as pd

def assign_plate_component(row):
    """
    Map a food directly from its nutrient profile (per 100 g)
    to a Healthy Eating Plate component.

    Components:
      - 'protein'    : fish, meat, poultry, eggs, dairy, legumes
      - 'vegetables' : non-starchy vegetables and fruit
      - 'carbs'      : grains and starchy vegetables
      - 'other'      : oils, nuts, sweets, mixed, etc.

    Expected columns in `row`:
      - energy_kcal_calculated
      - protein_g
      - carbs_g
      - fiber_g
      - fat_g
      - satfat_g
    """

    kcal    = float(row.get("energy_kcal_calculated", 0.0))
    protein = float(row.get("protein_g", 0.0))
    carbs   = float(row.get("carbs_g", 0.0))
    fiber   = float(row.get("fiber_g", 0.0))
    fat     = float(row.get("fat_g", 0.0))
    satfat  = float(row.get("satfat_g", 0.0))

    # Safe denominators
    kcal_safe  = max(kcal, 1e-6)
    carbs_safe = max(carbs, 1e-6)

    # Ratios of kcal coming from macros
    Pshare = 4 * protein / kcal_safe   # protein kcal share
    Cshare = 4 * carbs   / kcal_safe   # carb kcal share
    Fshare = 9 * fat     / kcal_safe   # fat kcal share

    # ----------------------------------------------------
    # 1) PROTEIN component
    #    - High protein density foods: meat, fish, poultry, eggs
    #    - Legumes: moderate protein + carbs + decent fiber
    # ----------------------------------------------------

    # High-protein, low-carb (animal protein style)
    protein_dense = (
        (protein >= 15 and carbs < 15) or   # classic meat/fish/eggs
        (Pshare >= 0.25 and carbs < 20)
    )

    # Legume-like profile (beans, lentils, chickpeas, etc.)
    legume_like = (
        5  <= protein <= 15 and
        10 <= carbs   <= 30 and
        fiber >= 3 and
        fat   < 15 and
        60 <= kcal <= 220
    )

    # Dairy-style: moderate protein + some carbs + some sat fat
    dairy_like = (
        protein >= 3 and
        carbs   >= 3 and
        satfat  >= 1.0 and
        kcal    >= 40
    )

    if protein_dense or legume_like or dairy_like:
        return "protein"

    # ----------------------------------------------------
    # 2) VEGETABLES component
    #    - Non-starchy vegetables
    #    - Fruit
    # ----------------------------------------------------

    # Non-starchy veg: low kcal, low–moderate carbs, some fiber
    nonstarchy_veg_like = (
        kcal < 80 and
        carbs < 15 and
        fiber >= 1.0
    )

    # Fruit: mostly carbs, low fat, some fiber, moderate kcal
    """fruit_like = (
        carbs >= 8 and
        fat < 5 and
        fiber >= 1.5 and
        kcal < 120
    )"""

    if nonstarchy_veg_like #or fruit_like:
        return "vegetables"

    # ----------------------------------------------------
    # 3) CARBS component
    #    - Grains and grain products
    #    - Starchy vegetables (potato, sweet potato, etc.)
    # ----------------------------------------------------

    # Grain / bread / pasta / rice style
    grain_like = (
        carbs >= 45 and
        kcal  >= 200 and
        fat   < 20
    )

    # Starchy veg: potatoes, yams, etc.
    starchy_veg_like = (
        15 <= carbs <= 35 and
        60 <= kcal <= 160 and
        fiber >= 1.5
    )

    if grain_like or starchy_veg_like:
        return "carbs"

    # ----------------------------------------------------
    # 4) Fallback
    #    Oils, nuts, sweets, mixed dishes, etc.
    # ----------------------------------------------------
    return "other"