

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

from langchain.chat_models import init_chat_model
from langchain.tools import tool


# ---------------------------------------------------------------------------
# Configuration & data loading
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
FOODS_CSV_PATH = os.getenv("FOODS_CSV_PATH", "../data/processed/food_with_clusters.csv")

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
    from the same cluster.

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
    cluster_value = food_row["cluster"]

    # Candidates: same cluster, but different item
    candidates = df[(df["cluster"] == cluster_value) & (~mask)]
    if candidates.empty:
        return (
            f"I found '{food_item}' in cluster {cluster_value}, "
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
        "kcal",
        "calories",
        "energy_kcal",
        "protein",
        "fiber",
        "sugars",
        "fat",
        "sat_fat",
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
        f"Original food: {food_row['food_item']} (cluster {cluster_value})\n"
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

def main():
    model = build_model()
    print("NutriMap tool-assisted LLM")
    print("Ask a question like: 'Suggest a healthier swap for chorizo from the dataset.'")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            user_input = input("Your question: ").strip()
            if not user_input:
                continue
            # First, ask the model to identify the main food item to swap
            extraction_prompt = (
                "You are helping with a nutrition tool. "
                "Given the following user message, identify the single main food item from "
                "the dataset that should be swapped. Respond with ONLY the food name, "
                "nothing else.\n\n"
                f"User message: {user_input}"
            )
            extraction_response = model.invoke(extraction_prompt)
            food_item = extraction_response.content.strip()

            # Then, call the CSV-based tool to suggest a swap
            swap_text = suggest_food_swap.invoke({"food_item": food_item})

            print("\n--- Suggestion ---")
            print(swap_text)
            print("------------------\n")
    except KeyboardInterrupt:
        print("\nExiting NutriMap tool-calling LLM.")


if __name__ == "__main__":
    main()
