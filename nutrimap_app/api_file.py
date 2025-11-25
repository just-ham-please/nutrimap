from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

#from package_folder.iris import my_prediction_function
from nutrimap_app.plate_optimizer import analyze_plate  # neue Plate-Analyse-Funktion


# ------------------------------------------------------------
# FastAPI instance
# ------------------------------------------------------------
app = FastAPI()


# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"greeting": "hello"}


# ------------------------------------------------------------
# Iris Demo Endpoint (Le Wagon Beispiel)
# ------------------------------------------------------------
@app.get("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
):
    """
    Simple demo endpoint using the iris model from package_folder.
    """
    prediction = my_prediction_function(
        sepal_length, sepal_width, petal_length, petal_width
    )
    return {"prediction": int(prediction[0])}


# ============================
# NutriMap Plate Analysis V1
# ============================

class PlateIngredient(BaseModel):
    role: str         # z.B. "protein", "carb", "veg", "fat"
    food_name: str    # exakter Name wie in foods_dummy.csv (food_item)
    grams: float      # Menge in Gramm


class PlateAnalysisRequest(BaseModel):
    ingredients: List[PlateIngredient]


class NutrientGap(BaseModel):
    actual: float
    target: float
    delta: float
    delta_pct: Optional[float]


class PlateAnalysisResponse(BaseModel):
    actual: Dict[str, float]              # z.B. {"energy_kcal": 550.0, ...}
    target: Dict[str, float]              # OPTIMAL_PLATE aus plate_optimizer.py
    gaps: Dict[str, NutrientGap]          # pro Nährstoff: Ist / Soll / Abweichung


@app.post("/plate/analyze", response_model=PlateAnalysisResponse)
def plate_analyze(input: PlateAnalysisRequest):
    """
    Nimmt eine Liste von Zutaten inkl. Grammangaben entgegen,
    berechnet die Nährwerte des Tellers und vergleicht ihn mit der Optimal-Plate.
    """
    # In dicts umwandeln, damit plate_optimizer unabhängig von Pydantic bleibt
    ingredients = [ing.dict() for ing in input.ingredients]

    result = analyze_plate(ingredients)
    # result ist bereits ein dict mit Schlüsseln: actual, target, gaps
    return result
