import streamlit as st
import requests
import pandas as pd
import altair as alt
from pathlib import Path
from nutrimap_app.llm_workflow import run_plate_workflow

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"   # Lokales Backend
# Für Deployment später z. B.:
# API_URL = "https://dein-backend-url"


st.set_page_config(page_title="NutriMap – Plate Analyzer", layout="centered")
st.title("NutriMap – Plate Analyzer 🍽️")


# ------------------------------------------------------------
# LOAD FOOD DATA
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "../data/processed/food_with_plate_roles.csv"

df = pd.read_csv(CSV_PATH)

df["food_item"] = df["food_item"].astype(str)
df["plate_role"] = df["plate_role"].astype(str)

roles = ["protein", "carbs", "fruit / veg", "fat"]

role_to_food_list = {
    role: sorted(df[df["plate_role"] == role]["food_item"].unique().tolist())
    for role in roles
}


# ------------------------------------------------------------
# UI: INGREDIENT SELECTION
# ------------------------------------------------------------

st.subheader("1️⃣ Select Ingredients")

cols = st.columns(4)

protein_item = cols[0].selectbox("Protein", role_to_food_list["protein"])
carb_item    = cols[1].selectbox("Carb",    role_to_food_list["carbs"])
veg_item     = cols[2].selectbox("Veg",     role_to_food_list["fruit / veg"])
fat_item     = cols[3].selectbox("Fat",     role_to_food_list["fat"])


st.subheader("2️⃣ Specify Amounts (grams)")

gcols = st.columns(4)

protein_g = gcols[0].number_input("Protein (g)", min_value=0, max_value=500, value=150, step=10)
carb_g    = gcols[1].number_input("Carb (g)",    min_value=0, max_value=500, value=120, step=10)
veg_g     = gcols[2].number_input("Veg (g)",     min_value=0, max_value=500, value=80,  step=10)
fat_g     = gcols[3].number_input("Fat (g)",     min_value=0, max_value=200, value=10,  step=5)

# Payload für Backend
payload = {
    "ingredients": [
        {"role": "protein", "food_name": protein_item, "grams": protein_g},
        {"role": "carb",    "food_name": carb_item,    "grams": carb_g},
        {"role": "veg",     "food_name": veg_item,     "grams": veg_g},
        {"role": "fat",     "food_name": fat_item,     "grams": fat_g},
    ]
}


# ------------------------------------------------------------
# API CALL + VISUALISIERUNG
# ------------------------------------------------------------

st.subheader("3️⃣ Analyze your plate")

if st.button("Analyze Plate"):
    st.write("Sending request...")

    try:
        resp = requests.post(f"{API_URL}/plate/analyze", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        st.success("Plate analyzed successfully!")

        actual = data["actual"]
        target = data["target"]
        gaps   = data["gaps"]

        # ----------------------------------------------------
        # A) Prozent des Optimalwerts (0–200 % Skala)
        # ----------------------------------------------------
        st.subheader("📊 Coverage vs. Optimal (in %)")

        percent_rows = []
        for nutrient, a_val in actual.items():
            t_val = target.get(nutrient, 0)
            if t_val and t_val != 0:
                pct = a_val / t_val * 100
                percent_rows.append({
                    "nutrient": nutrient,
                    "percent_of_optimal": pct
                })

        percent_df = pd.DataFrame(percent_rows)

        if not percent_df.empty:
            chart = (
                alt.Chart(percent_df)
                .mark_bar()
                .encode(
                    x=alt.X("nutrient:N", title="Nutrient"),
                    y=alt.Y(
                        "percent_of_optimal:Q",
                        title="% of optimal",
                        scale=alt.Scale(domain=[0, 200])  # 0–200 % Anzeige
                    ),
                    tooltip=["nutrient", "percent_of_optimal"]
                )
                .properties(height=300)
            )

            # Referenzlinie bei 100 %
            rule = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")

            st.altair_chart(chart + rule, use_container_width=True)
        else:
            st.info("No nutrients with non-zero targets found for percentage chart.")

        # ----------------------------------------------------
        # B) Tabelle: Actual, Optimal, Abweichungen
        # ----------------------------------------------------
        st.subheader("📋 Nutrient Details")

        table_rows = []
        for nutrient, gap in gaps.items():
            table_rows.append({
                "Nutrient": nutrient,
                "Actual": gap["actual"],
                "Optimal": gap["target"],
                "Δ abs": gap["delta"],
                "Δ %": gap["delta_pct"],
            })

        table_df = pd.DataFrame(table_rows)
        table_df = table_df.set_index("Nutrient")

        st.dataframe(table_df)

        # ----------------------------------------------------
        # C) Debug ganz unten
        # ----------------------------------------------------
        st.subheader("🛠 Debug: Raw API Response")
        st.json(data)

        # ----------------------------------------------------
        # D) Suggestions from LLM
        # ----------------------------------------------------
        st.subheader("4️⃣ Suggestions")

        try:
            # Use the same ingredients the user selected to drive the LLM workflow.
            llm_response = run_plate_workflow(payload["ingredients"])

            # Nicely formatted LLM output
            st.markdown("#### Personalized nutrition advice")
            st.markdown(llm_response)
        except Exception as e:
            st.error(f"Error while generating suggestions: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
