

"""
Tool-calling LLM script for NutriMap.

This script:
- Loads a CSV file with foods, nutritional values, and clusters.
- Exposes a LangChain tool `suggest_food_swap` that looks up a food in the dataset
  and suggests a viable swap from the same cluster.
- Creates a tool-calling agent with Gemini 2.0 Flash (via `init_chat_model`).
- Reads a natural-language question from stdin and prints the agent response.

You can control some settings via environment variables:
  GEMINI_MODEL_NAME      (default: "gemini-2.0-flash-exp")
  GEMINI_TEMPERATURE     (default: "0.3")
  GEMINI_MAX_OUTPUT_TOKENS (default: "512")
  FOODS_CSV_PATH         (default: "NUTRIMAP/data/processed/food_with_clusters.csv")
"""

import os
from dotenv import load_dotenv
import pandas as pd
import requests
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain.tools import tool

# For local development - API_URL to get access to the responses from the front_end
API_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Configuration & data loading
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
FOODS_CSV_PATH = os.getenv("FOODS_CSV_PATH", "./data/processed/food_with_plate_roles.csv")

# Load food dataset once at startup
if not os.path.exists(FOODS_CSV_PATH):
    raise FileNotFoundError(
        f"CSV file not found at '{FOODS_CSV_PATH}'. "
        "Set the FOODS_CSV_PATH env var or create the expected file."
    )

FOODS_DF = pd.read_csv(FOODS_CSV_PATH)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def suggest_food_swap(food_item: str) -> str:
    """
    Given a food name from the dataset, suggest a healthier swap
    from the same plate role.

    The CSV is expected to contain at least:
    - 'food_item': name of the food
    - 'cluster': cluster/group label
    - optional numeric columns like 'kcal', 'sugars', 'fat', 'protein', 'fiber'

    Adjust the column names in this function to match your dataset.
    """
    df = FOODS_DF

    if "food_item" not in df.columns or "cluster" not in df.columns:
        return (
            "The dataset must contain 'food_item' and 'cluster' columns. "
            "Please update the tool implementation to match your CSV schema."
        )

    # Case-insensitive match of the requested food
    mask = df["food_item"].astype(str).str.lower() == food_item.lower()
    if not mask.any():
        available = ", ".join(sorted(df["food_item"].astype(str).head(20)))
        return (
            f"I couldn't find '{food_item}' in the dataset.\n"
            f"Example foods I do know: {available}"
        )

    food_row = df[mask].iloc[0]
    plate_role_value = food_row["plate_role"]

    # Candidates: same plate role, but different item
    candidates = df[(df["plate_role"] == plate_role_value) & (~mask)]
    if candidates.empty:
        return (
            f"I found '{food_item}' in cluster {plate_role_value}, "
            "but there are no alternative items in that cluster."
        )

    # Simple heuristic: prefer lower kcal and sugars if those columns exist
    sort_columns = []
    for col in ["kcal", "calories", "energy_kcal"]:
        if col in candidates.columns:
            sort_columns.append(col)
            break
    for col in ["sugars", "sugars_g"]:
        if col in candidates.columns:
            sort_columns.append(col)
            break

    if sort_columns:
        candidates = candidates.sort_values(sort_columns, ascending=True)

    swap_row = candidates.iloc[0]

    # Build a short comparison string using any nutrition columns that exist
    def format_macro(row, col):
        return f"{col}={row[col]}" if col in row.index else None

    possible_cols = [
        'food_item',
        'fat_g',
        'satfat_g',
        'carbs_g',
        'protein_g',
        'fiber_g',
        'energy_kcal_calculated',
        'cluster',
        'subcluster',
        'plate_role'
    ]
    original_macros = [
        m for col in possible_cols if (m := format_macro(food_row, col)) is not None
    ]
    swap_macros = [
        m for col in possible_cols if (m := format_macro(swap_row, col)) is not None
    ]

    original_desc = (
        ", ".join(original_macros) if original_macros else "no detailed macros available"
    )
    swap_desc = (
        ", ".join(swap_macros) if swap_macros else "no detailed macros available"
    )

    return (
        f"Original food: {food_row['food_item']} (cluster {plate_role_value})\n"
        f"Nutrition: {original_desc}\n\n"
        f"Suggested swap: {swap_row['food_item']} (same cluster)\n"
        f"Nutrition: {swap_desc}\n\n"
        "Use this suggestion as a starting point. You can refine it based on the patient's "
        "specific goals and preferences."
    )


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_model():
    """
    Create a chat model (Gemini 2.0 Flash) configured with the system message.
    """
    model = init_chat_model(
        model=GEMINI_MODEL_NAME,
        model_provider="google_genai",
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        system_message=(
            "You are a professional nutritionist who communicates clearly and compassionately "
            "with patients. You help them find practical and healthier food swaps. "
            "When appropriate, you will be asked to identify the main food item that should be "
            "swapped, based on the user's description."
        ),
    )
    return model


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plate/LLM workflow helpers
# ---------------------------------------------------------------------------

def call_plate_analyze(ingredients: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call the FastAPI /plate/analyze endpoint and return the JSON result.

    Parameters
    ----------
    ingredients: list of dicts with keys ["role", "food_name", "grams"].
    """
    payload = {"ingredients": ingredients}
    response = requests.post(f"{API_URL}/plate/analyze", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def build_prompt_from_plate(plate_result: Dict[str, Any]) -> str:
    """Turn the plate analysis JSON into a natural-language prompt for the LLM.

    The prompt focuses on the main nutrients and highlights where the plate is
    over or under the target values. It returns a single string that can be
    passed as the user message to the model.
    """
    actual = plate_result.get("actual", {})
    target = plate_result.get("target", {})
    gaps = plate_result.get("gaps", {})

    # Build a compact human-readable summary, focusing on up to 4 biggest gaps
    # (by absolute percentage difference) if available.
    gap_items = []
    for nutrient, info in gaps.items():
        delta_pct = info.get("delta_pct")
        if delta_pct is None:
            continue
        gap_items.append((nutrient, abs(delta_pct), info))

    gap_items.sort(key=lambda x: x[1], reverse=True)
    top_gaps = gap_items[:4] if gap_items else []

    summary_lines = ["Here is the analysis of the patient's plate:"]

    if top_gaps:
        summary_lines.append("The 4 nutrients with the largest deviations from the target are:")
        for nutrient, _abs_delta_pct, info in top_gaps:
            summary_lines.append(
                f"- {nutrient}: actual {info.get('actual')} vs target {info.get('target')} "
                f"(delta {info.get('delta')}, {info.get('delta_pct')}% from target)"
            )
    else:
        # Fallback: list all nutrients if for some reason we do not have deltas
        for nutrient, value in actual.items():
            t = target.get(nutrient)
            summary_lines.append(f"- {nutrient}: actual {value}, target {t}")

    summary_lines.append(
        "Based on this plate analysis, suggest concrete nutrition advice and, "
        "if useful, identify one main food item that should be swapped for a "
        "healthier alternative. Focus on practical, patient-friendly guidance."
    )

    return "\n".join(summary_lines)


def run_plate_workflow(ingredients: List[Dict[str, Any]]) -> str:
    """High-level function to be called from Streamlit.

    1. Calls the NutriMap FastAPI /plate/analyze endpoint with the given
       ingredients.
    2. Builds a natural-language prompt from the analysis.
    3. Sends the prompt to the LLM and returns the model's answer as a string.

    This keeps all the LLM-related logic in one place, while Streamlit only
    needs to pass the ingredients list.
    """
    # 1) Call the API
    plate_result = call_plate_analyze(ingredients)

    # 2) Turn result into a user-style prompt
    user_prompt = build_prompt_from_plate(plate_result)

    # 3) Ask the model for advice. For now we use a simple single-turn call.
    model = build_model()
    response = model.invoke(user_prompt)

    # Many LangChain chat models return an object with a `.content` attribute
    # that holds the main text.
    return getattr(response, "content", str(response))


# ---------------------------------------------------------------------------
# CLI entrypoint (for local testing)
# ---------------------------------------------------------------------------

def main() -> None:
    """Simple CLI demo for local testing.

    In production, Streamlit should call `run_plate_workflow(...)` directly
    and display the returned text to the user.
    """
    # Example ingredients; replace with real data in your Streamlit app.
    example_ingredients: List[Dict[str, Any]] = [
        {"role": "protein", "food_name": "Chicken Breast", "grams": 150},
        {"role": "carb", "food_name": "White Rice", "grams": 120},
        {"role": "veg", "food_name": "Broccoli", "grams": 80},
    ]

    advice = run_plate_workflow(example_ingredients)
    print("NutriMap LLM advice:\n")
    print(advice)


if __name__ == "__main__":
    main()
