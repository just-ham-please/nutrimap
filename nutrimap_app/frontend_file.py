import streamlit as st
from pathlib import Path
import requests
import time
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="NutriMap Demo", layout="centered")

# --- GREEN BUTTON ONLY (NO GLOBAL BACKGROUND/TEXT CHANGES) ---
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #22a34f;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.4rem 0.9rem;
    }
    .stButton > button:hover {
        background-color: #1b7c3a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = "https://api-nutrimap-1002154750813.europe-west1.run.app/"

st.title("NutriMap Demo")

st.markdown(
    "Below you find a small demo: on top the old flower test model, "
    "followed by the first NutriMap V1 ingredient selector."
)

# ==========================================
# DEV / DEBUG SECTION – OLD FLOWER MODEL
# ==========================================
with st.expander("🔬 Dummy Model – Flowers (dev only)"):
    st.caption("Legacy Iris model for API testing. Not part of the NutriMap UI.")

    sepal_length = st.slider("Select a value A", min_value=0, max_value=4, value=1, step=1)
    sepal_width = st.slider("Select a value B", min_value=0, max_value=4, value=1, step=1)
    petal_length = st.slider("Select a value C", min_value=0, max_value=4, value=1, step=1)
    petal_width = st.slider("Select a value D", min_value=0, max_value=4, value=1, step=1)

    url = f"{API_URL}/predict"
    params = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        response = requests.get(url, params=params, timeout=3).json()
        st.success(f"This flower belongs to category **{str(response['prediction'])}**")
    except Exception:
        st.warning("Backend not reachable – expected for this demo.")

# ==========================================
# MAIN SECTION – NUTRIMAP V1
# ==========================================
st.markdown("---")
st.header("🍽️ NutriMap – V1 Prototype")
st.write("Select your main ingredients from the dropdowns below:")

# Determine absolute path to this file
BASE_DIR = Path(__file__).resolve().parent  # /nutrimap/nutrimap_app

# --- load UI data (dummy for now, later real cleaned CSV) ---
CSV_PATH = BASE_DIR.parent / "test_data_justus" / "foods_ui_dummy.csv"
df_ui = pd.read_csv(CSV_PATH)

# --- split by plate_role ---
protein_list = (
    df_ui[df_ui["plate_role"] == "protein"]["food_name"]
    .dropna()
    .sort_values()
    .unique()
    .tolist()
)

carb_list = (
    df_ui[df_ui["plate_role"] == "carb"]["food_name"]
    .dropna()
    .sort_values()
    .unique()
    .tolist()
)

fat_list = (
    df_ui[df_ui["plate_role"] == "fat"]["food_name"]
    .dropna()
    .sort_values()
    .unique()
    .tolist()
)


# Layout: left = inputs + button, right = summary & suggestion
col_left, col_right = st.columns([2, 1])

with col_left:
    protein_option = st.selectbox("🥩 Protein source", protein_list)
    carb_option = st.selectbox("🍞 Carb source", carb_list)
    fat_option = st.selectbox("🥑 Fat source", fat_list)

    calculate = st.button("Calculate better alternatives")

with col_right:
    st.subheader("Your dish")
    st.markdown(
        f"- **Protein:** {protein_option}  \n"
        f"- **Carbs:** {carb_option}  \n"
        f"- **Fats:** {fat_option}"
    )
    if calculate:
        payload = {
            "protein": protein_option,
            "carb": carb_option,
            "fat": fat_option
        }

        with st.spinner("Calculating better alternatives..."):
            time.sleep(1.0)  # optional Demo-Delay
            try:
                response = requests.post(
                    f"{API_URL.rstrip('/')}/optimize_plate",
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                st.subheader("Suggested improvement")
                st.success(
                    f"Better alternatives could be **{data['better_protein']}** as protein, "
                    f"**{data['better_carb']}** as carbs and **{data['better_fat']}** as fats."
                )
            except Exception as e:
                st.error(f"Optimization failed: {e}")

st.markdown("---")
st.caption("NutriMap V1 prototype – backend not connected yet.")
