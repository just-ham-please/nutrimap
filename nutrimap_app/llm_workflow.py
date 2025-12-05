

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
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
import requests
from typing import List, Dict, Any
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool

# For local development - API_URL to get access to the responses from the front_end
#API_URL = "http://127.0.0.1:8080"

import os

API_URL = os.getenv(
    "API_URL",
    "https://nutrimap-backend-1002154750813.europe-west1.run.app",
)


# ---------------------------------------------------------------------------
# Configuration & data loading
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512"))
ROOT = Path(__file__).resolve().parent.parent
FOODS_CSV_PATH = ROOT / "data" / "processed" / "food_with_plate_roles.csv"

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

    # Normalize and lightly clean the incoming food item string
    food_item_clean = str(food_item).strip().lower()
    # If the model accidentally includes extra info like ", grams=120", strip that off
    for sep in [" grams=", ", grams=", "(grams", " g ", " grams "]:
        if sep in food_item_clean:
            food_item_clean = food_item_clean.split(sep)[0].strip()
            break

    # Case-insensitive match of the requested food (after cleaning)
    names_lower = df["food_item"].astype(str).str.lower()
    mask = names_lower == food_item_clean

    # If no exact match, try a more forgiving contains-based match
    if not mask.any():
        mask = names_lower.str.contains(food_item_clean, na=False)

    if not mask.any():
        available = ", ".join(sorted(df["food_item"].astype(str).head(20)))
        return (
            f"I couldn't find '{food_item}' in the dataset after normalization "
            f"('{food_item_clean}').\n"
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
    model_with_tools = model.bind_tools([suggest_food_swap])
    return model_with_tools


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


def build_prompt_from_plate(plate_result: Dict[str, Any],
    ingredients: List[Dict[str, Any]],) -> str:
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
        "Provide only one suggestion per food_item. If there are multiple."
    )
    # Telling the model exactly which foods are on the plate
    summary_lines.append(
        "\nHere are the foods on the plate. Each food_name appears in single quotes. "
        "When you call the tool, you must copy exactly one of the quoted food_name strings "
        "as the `food_item` argument."
    )
    for ing in ingredients:
        summary_lines.append(
            f"- food_name='{ing['food_name']}')"
        )

    summary_lines.append(
        "\nYou must always call the `suggest_food_swap` tool exactly once, "
        "using one of the `food_name` values above as the `food_item` argument. "
        "Do not ask the user for a food; pick one yourself."
    )

    summary_lines.append(
        "When you propose swaps, they must make culinary sense: "
        "swap a grain for another grain that could realistically replace it on the plate "
        "(e.g. white rice → brown rice, quinoa, bulgur), not flour or raw ingredients. "
        "Swap a protein for another cooked protein (e.g. chicken → tofu, fish, legumes), "
        "not something from a completely different role. "
        "You may suggest more than one swap (up to 3) if that helps balance the plate."
    )

    summary_lines.append(
        "When you name the swapped-in food, use the exact `food_item` string from the database "
        "(see the tool results), not a simplified or generic name."
    )

    summary_lines.append(
        "\nYour response MUST follow this structure:\n"
        "1. 2–3 sentences summarizing whether the plate is balanced, "
        "   referring to the over/under nutrients above.\n"
        "2. Clearly name one `food_name` from the list that you are swapping out.\n"
        "3. Use the `suggest_food_swap` tool to get a concrete alternative.\n"
        "4. In 2–3 sentences, explain why this swap helps with the specific "
        "   nutrient gaps (e.g., lowers fat, increases fiber, etc.)."
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
    user_prompt = build_prompt_from_plate(plate_result, ingredients)

    # 3) Ask the model for advice. For now we use a simple single-turn call.
    model = build_model()
    messages = [HumanMessage(content=user_prompt)]
    response = model.invoke(messages)

    # If the model wants to call a tool, handle it
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            if tc["name"] == "suggest_food_swap":
                food_item = tc["args"]["food_item"]
                tool_output = suggest_food_swap.invoke({"food_item": food_item})
                messages.append(response)
                messages.append(
                    HumanMessage(content=f"Tool result for {food_item}:\n{tool_output}")
                )
                final = model.invoke(messages)
                return getattr(final, "content", str(final))

    # Fallback: no tool usage
    return getattr(response, "content", str(response))


# Function that extracts the food_item from the suggestion llm response.
def suggestion_to_csv(response: str) -> str:
    '''Build a prompt that asks the LLM to extract swap foods and grams from the
    first LLM suggestion, and return them as a tiny CSV with columns:
    food_name,grams.

    The downstream code (`run_post_suggestion_prompt`) will parse this CSV and
    turn it into a list of ingredient dicts:
        [{"role": "...", "food_name": "...", "grams": ...}, ...]
    which is what `compute_plate_nutrients` expects.
    '''
    # We include a snapshot of the foods table so the LLM can align names to
    # real entries in the database, but here we only care about extracting
    # `food_name` and `grams`.
    preferred_columns = [
        "food_item",
        "plate_role",
        "energy_kcal_calculated",
        "fat_g",
        "satfat_g",
        "carbs_g",
        "protein_g",
        "fiber_g",
    ]
    available_columns = [c for c in preferred_columns if c in FOODS_DF.columns]
    csv_snapshot = FOODS_DF[available_columns].to_csv(index=False)

    prompt = f"""You are a data extraction assistant helping a nutrition app.

        You are given:
        1. A piece of nutrition advice produced by another model. This text may include
        a section with tool output, for example lines like:
            "Original food: ..."
            "Suggested swap: Flour, buckwheat ..."
        2. A CSV table describing foods and their nutritional information. The primary
        key column in this table is called `food_item`.

        Your task:
        - Read the nutrition advice carefully.
        - Identify all foods that are explicitly mentioned as **swapped-in** foods on
        the new plate (i.e., foods that the patient should eat instead of the
        original items). In particular, look for lines starting with phrases such as
        "Suggested swap:" and use the food names that appear there.
        - For each such food, find the closest matching `food_item` in the CSV snapshot.
        You MUST choose the name from the `food_item` column, not invent new names.
        Copy the `food_item` string exactly as written (including capitalization,
        commas, and spaces).
        - For each selected `food_item`, decide how many grams of that food will be on
        the plate. If the advice does not clearly specify the amount, choose a
        reasonable default such as 100 grams.
        - Return ONLY a CSV with the header row:
            food_name,grams
        followed by one row per suggested food.

        Requirements:
        - The `food_name` column must contain only values that exactly match a
        `food_item` from the CSV table.
        - The `grams` column must be a number (no units or words).
        - Do not output any markdown, explanation, commentary, or quotes—only raw CSV.

        Nutrition advice:
        \"\"\"{response}\"\"\"

        CSV table (schema and data to reference; you do not need to repeat it):
        {csv_snapshot}

        Remember: your final answer must be only the CSV with columns:
        food_name,grams
        and nothing else.
        """

    return prompt

# Function that runs the extraction of suggestion from suggestion reposone
def run_post_suggestion_prompt(response: str) -> Dict[str, List[Dict[str, Any]]]:
    """Second-stage LLM call that turns a free-text suggestion into a structured
    list of ingredients compatible with `compute_plate_nutrients`.

    Expected general format (per line):
        <food_name>,<grams>

    Returned format:
        {"food_swap_list": [{"role": "...", "food_name": "...", "grams": ...}, ...]}
    """
    model = build_model()
    prompt = suggestion_to_csv(response)
    post_suggestion_result = model.invoke(prompt)

    csv_text = getattr(post_suggestion_result, "content", str(post_suggestion_result)).strip()
    if not csv_text:
        return {"food_swap_list": []}

    # --- Manual CSV parsing, tolerant to headers and commas in names ---
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if not lines:
        return {"food_swap_list": []}

    rows = []
    for line in lines:
        # Split from the right: everything before last comma = name, last part = grams
        try:
            name_part, grams_part = line.rsplit(",", 1)
        except ValueError:
            # Cannot split into two parts -> skip
            continue

        name_part = name_part.strip()
        grams_part = grams_part.strip()

        if not name_part or not grams_part:
            continue

        # Try to interpret grams as a float; header lines will fail here
        try:
            grams_val = float(grams_part)
        except ValueError:
            # Not numeric (likely a header row) -> skip
            continue

        rows.append({"food_name": name_part, "grams": grams_val})

    if not rows:
        return {"food_swap_list": []}

    df_swaps = pd.DataFrame(rows)
    # --- End of manual CSV parser ---

    def resolve_to_food_item(name: str) -> Dict[str, Any] | None:
        """Return canonical food_item and plate_role from FOODS_DF, or None if not found."""
        if "food_item" not in FOODS_DF.columns or "plate_role" not in FOODS_DF.columns:
            return None

        names_lower = FOODS_DF["food_item"].astype(str).str.lower()
        q = str(name).strip().lower()

        # Exact match
        exact_mask = names_lower == q
        if exact_mask.any():
            row = FOODS_DF[exact_mask].iloc[0]
            return {"food_item": row["food_item"], "plate_role": row["plate_role"]}

        # Fuzzy "contains" match
        contains_mask = names_lower.str.contains(q, na=False)
        if contains_mask.any():
            row = FOODS_DF[contains_mask].iloc[0]
            return {"food_item": row["food_item"], "plate_role": row["plate_role"]}

        return None

    food_swap_list: List[Dict[str, Any]] = []
    for _, row in df_swaps.iterrows():
        raw_name = str(row["food_name"])
        grams_val = float(row["grams"])

        resolved = resolve_to_food_item(raw_name)
        if resolved is None:
            continue

        food_swap_list.append(
            {
                "role": str(resolved["plate_role"]),
                "food_name": str(resolved["food_item"]),
                "grams": grams_val,
            }
        )

    return {"food_swap_list": food_swap_list}




# ---------------------------------------------------------------------------
# CLI entrypoint (for local testing)
# ---------------------------------------------------------------------------

def main() -> None:
    """Simple CLI demo for local testing.

    In production, Streamlit should call `run_plate_workflow(...)` directly
    and display the returned text to the user.
    """
    # Build a simple example plate using the first few foods from FOODS_DF so that
    # the names and roles are guaranteed to exist in the dataset.
    example_ingredients: List[Dict[str, Any]] = []
    role_column = "plate_role" if "plate_role" in FOODS_DF.columns else "role"
    for _, row in FOODS_DF.head(3).iterrows():
        example_ingredients.append(
            {
                "role": row.get(role_column, "other"),
                "food_name": row["food_item"],
                "grams": 100.0,
            }
        )

    advice = run_plate_workflow(example_ingredients)
    print("NutriMap LLM advice:\n")
    print(advice)


if __name__ == "__main__":
    main()
