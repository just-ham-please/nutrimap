from fastapi import FastAPI
import pickle

from package_folder.iris import my_prediction_function


from plate_optimizer import optimize_plate  # out optimization function

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"hello"}

# Prediction endpoint
@app.get("/predict")
def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Use the function in our package to run the prediction
    prediction = my_prediction_function(sepal_length, sepal_width, petal_length, petal_width)

    # Return prediction
    return {"prediction": int(prediction[0])}


# ============================
# NutriMap V1 Endpoint
# ============================

class PlateInput(BaseModel):
    protein: str
    carb: str
    fat: str

class PlateOutput(BaseModel):
    better_protein: str
    better_carb: str
    better_fat: str


@app.post("/optimize_plate", response_model=PlateOutput)
def optimize(input: PlateInput):
    """
    Receives selected ingredients and returns better alternatives.
    """
    result = optimize_plate(
        protein=input.protein,
        carb=input.carb,
        fat=input.fat
    )
    return result
