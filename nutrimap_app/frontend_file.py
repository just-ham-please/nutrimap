import streamlit as st
import requests
import pandas as pd
import altair as alt
from pathlib import Path
from llm_workflow import run_plate_workflow, run_post_suggestion_prompt
from plate_optimizer import post_suggestion_comparison


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

#API_URL = "http://127.0.0.1:8080"   # Lokales Backend
# For Deployment
API_URL = "https://nutrimap-backend-1002154750813.europe-west1.run.app"


st.set_page_config(page_title="NutriMap – Eat Better. Feel Better.", layout="centered")

st.markdown("""
<style>
/* Primary button leicht anpassen (optional, weil Theme schon greift) */
.stButton > button {
    background-color: #22A34F;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.4rem 0.9rem;
}
.stButton > button:hover {
    background-color: #1B7C3A;
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.title("NutriMap")
st.subheader("Analyze and Optimize Your Plate 🍽️")

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

st.subheader("1️⃣ Build your plate")
st.markdown("Pick one main food per role and an approximate portion size in grams.")

def role_row(icon, title, subtitle, options, default_grams, key_prefix):
    st.markdown(f"{icon} **{title}**")
    st.caption(subtitle)

    col1, col2 = st.columns([2.5, 1])
    with col1:
        item = st.selectbox(
            "Food",
            options,
            index=None,
            placeholder=f"Select {title.lower()}...",
            key=f"{key_prefix}_food",
        )
    with col2:
        grams = st.number_input(
            "Amount (g)",
            min_value=0,
            max_value=500,
            value=default_grams,
            step=10,
            key=f"{key_prefix}_grams",
        )

    return item, grams


protein_item, protein_g = role_row(
    "🥩",
    "Protein",
    "Main protein of your meal.",
    role_to_food_list["protein"],
    default_grams=150,
    key_prefix="protein",
)

st.markdown("<hr style='margin:1.2rem 0; border:0; border-top:1px solid #D1D5DB;'>", unsafe_allow_html=True)


carb_item, carb_g = role_row(
    "🍚",
    "Carbs",
    "Main carb or starch on your plate.",
    role_to_food_list["carbs"],
    default_grams=120,
    key_prefix="carb",
)

st.markdown("<hr style='margin:1.2rem 0; border:0; border-top:1px solid #D1D5DB;'>", unsafe_allow_html=True)


veg_item, veg_g = role_row(
    "🥦",
    "Veg & Fruit",
    "Non-starchy vegetables or fruit.",
    role_to_food_list["fruit / veg"],
    default_grams=80,
    key_prefix="veg",
)

st.markdown("<hr style='margin:1.2rem 0; border:0; border-top:1px solid #D1D5DB;'>", unsafe_allow_html=True)


fat_item, fat_g = role_row(
    "🥜",
    "Fats & Extras",
    "Oils, nuts, seeds or dressings.",
    role_to_food_list["fat"],
    default_grams=10,
    key_prefix="fat",
)

st.markdown("<hr style='margin:1.2rem 0; border:0; border-top:1px solid #D1D5DB;'>", unsafe_allow_html=True)



# Payload for Backend
payload = {
    "ingredients": [
        {"role": "protein", "food_name": protein_item, "grams": protein_g},
        {"role": "carb",    "food_name": carb_item,    "grams": carb_g},
        {"role": "veg",     "food_name": veg_item,     "grams": veg_g},
        {"role": "fat",     "food_name": fat_item,     "grams": fat_g},
    ]
}


# ------------------------------------------------------------
# API CALL + VISUALS
# ------------------------------------------------------------

st.subheader("2️⃣ Analyze your plate")

if st.button("Analyze Plate"):
    # Basic validation: all roles must have a selected food
    if any(x is None for x in [protein_item, carb_item, veg_item, fat_item]):
        st.error("Please select a food for each role before analyzing your plate.")
    else:
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
            # A) Nutrient gaps vs optimal (in grams)
            # ----------------------------------------------------
            st.subheader("📊 Nutrient gaps vs optimal (g)")

            gap_rows = []
            for nutrient, gap in gaps.items():
                delta = gap["delta"]  # positive = above optimal, negative = below
                gap_rows.append(
                    {
                        "nutrient": nutrient,
                        "delta_g": delta,
                        "actual": gap["actual"],
                        "optimal": gap["target"],
                    }
                )

            gap_df = pd.DataFrame(gap_rows)

            if not gap_df.empty:
                # Split kcal and non-kcal nutrients
                gap_df["abs_delta"] = gap_df["delta_g"].abs()
                is_kcal = gap_df["nutrient"].str.contains("kcal", case=False, na=False)

                non_kcal_df = gap_df[~is_kcal]
                kcal_df = gap_df[is_kcal]

                # Chart for non-kcal nutrients (grams)
                if not non_kcal_df.empty:
                    chart = (
                        alt.Chart(non_kcal_df)
                        .mark_bar()
                        .encode(
                            y=alt.Y(
                                "nutrient:N",
                                sort=alt.SortField(field="abs_delta", order="descending"),
                                title="Nutrient",
                            ),
                            x=alt.X(
                                "delta_g:Q",
                                title="Difference vs optimal (g)",
                                scale=alt.Scale(domain=[-50, 50]),
                            ),
                            color=alt.value("#22A34F"),
                            tooltip=[
                                alt.Tooltip("nutrient:N", title="Nutrient"),
                                alt.Tooltip("actual:Q", title="Actual"),
                                alt.Tooltip("optimal:Q", title="Optimal"),
                                alt.Tooltip("delta_g:Q", title="Δ vs optimal (g)"),
                            ],
                        )
                        .properties(height=300)
                    )

                    zero_line = (
                        alt.Chart(pd.DataFrame({"x": [0]}))
                        .mark_rule(strokeDash=[4, 4])
                        .encode(x="x:Q")
                    )

                    st.altair_chart(chart + zero_line, use_container_width=True)
                else:
                    st.info("No non-kcal nutrient gaps to display.")

                # Separate handling for kcal if present
                if not kcal_df.empty:
                    st.markdown("#### Energy (kcal)")
                    for _, row in kcal_df.iterrows():
                        st.write(
                            f"**{row['nutrient']}** – Actual: {row['actual']:.0f} kcal, "
                            f"Optimal: {row['optimal']:.0f} kcal, "
                            f"Δ: {row['delta_g']:.0f} kcal"
                        )
            else:
                st.info("No nutrient gaps to display.")


            # if not gap_df.empty:
            #     # Sort nutrients by absolute gap, biggest first
            #     gap_df["abs_delta"] = gap_df["delta_g"].abs()

            #     chart = (
            #         alt.Chart(gap_df)
            #         .mark_bar()
            #         .encode(
            #             y=alt.Y(
            #                 "nutrient:N",
            #                 sort=alt.SortField(field="abs_delta", order="descending"),
            #                 title="Nutrient",
            #             ),
            #             x=alt.X(
            #                 "delta_g:Q",
            #                 title="Difference vs optimal (g)",
            #                 scale=alt.Scale(zero=True),
            #             ),
            #             color=alt.condition(
            #                 alt.datum.delta_g > 0,
            #                 alt.value("#22A34F"),  # above optimal
            #                 alt.value("#EF4444"),  # below optimal
            #             ),
            #             tooltip=[
            #                 alt.Tooltip("nutrient:N", title="Nutrient"),
            #                 alt.Tooltip("actual:Q", title="Actual"),
            #                 alt.Tooltip("optimal:Q", title="Optimal"),
            #                 alt.Tooltip("delta_g:Q", title="Δ vs optimal (g)"),
            #             ],
            #         )
            #         .properties(height=300)
            #     )

            #     zero_line = (
            #         alt.Chart(pd.DataFrame({"x": [0]}))
            #         .mark_rule(strokeDash=[4, 4])
            #         .encode(x="x:Q")
            #     )

            #     st.altair_chart(chart + zero_line, use_container_width=True)
            # else:
            #     st.info("No nutrient gaps to display.")

            # ----------------------------------------------------
            # B) Table: Actual, Optimal, Deviations
            # ----------------------------------------------------
            st.subheader("📋 Nutrient details")

            table_rows = []
            for nutrient, gap in gaps.items():
                table_rows.append(
                    {
                        "Nutrient": nutrient,
                        "Actual": gap["actual"],
                        "Optimal": gap["target"],
                        "Δ abs": gap["delta"],
                        "Δ %": gap["delta_pct"],
                    }
                )

            table_df = pd.DataFrame(table_rows).set_index("Nutrient")
            st.dataframe(table_df)

            # ----------------------------------------------------
            # C) Suggestions from LLM
            # ----------------------------------------------------
            st.subheader("4️⃣ Suggestions")

            try:
                # Use the same ingredients the user selected to drive the LLM workflow.
                llm_response = run_plate_workflow(payload["ingredients"])

                st.markdown("#### Personalized nutrition advice")
                st.markdown(llm_response)
            except Exception as e:
                st.error(f"Error while generating suggestions: {e}")
                llm_response = None  # so we do not crash below

            # ----------------------------------------------------
            # D) Food swaps visualization from LLM
            # ----------------------------------------------------
            st.subheader("5️⃣ Food Swap – Nutrient comparison")

            try:
                if not llm_response:
                    st.info("No LLM response available to derive food swaps.")
                else:
                    # 1) Let the LLM workflow extract the swapped foods from its own suggestion text
                    post_suggestion_result = run_post_suggestion_prompt(llm_response)

                    st.markdown("#### Food swap comparison")

                    food_swaps = post_suggestion_result.get("food_swap_list", [])

                    if not food_swaps:
                        st.info("No swap foods were suggested.")
                    else:
                        st.markdown("**Suggested swap food(s):**")
                        for swap in food_swaps:
                            st.markdown(
                                f"- `{swap['food_name']}` "
                                f"(role: {swap['role']}, grams: {swap['grams']})"
                            )

                    # 2) Compute nutrients for the suggested swap
                    swap_comparison = post_suggestion_comparison(post_suggestion_result)

                    gaps_swap = swap_comparison["gaps_swap"]

                    swap_rows = []
                    for nutrient, gap in gaps_swap.items():
                        swap_rows.append(
                            {
                                "Nutrient": nutrient,
                                "Actual (swap)": gap["actual"],
                                "Optimal": gap["target"],
                                "Δ abs (swap)": gap["delta"],
                                "Δ % (swap)": gap["delta_pct"],
                            }
                        )

                    swap_df = pd.DataFrame(swap_rows).set_index("Nutrient")
                    st.dataframe(swap_df)

                    # ----------------------------------------------------
                    # E) Nutrient gaps after swap (in grams)
                    # ----------------------------------------------------
                    st.subheader("📊 Nutrient gaps after swap (g)")

                    swap_gap_rows = []
                    for nutrient, gap in gaps_swap.items():
                        delta = gap["delta"]  # positive = above optimal, negative = below
                        swap_gap_rows.append(
                            {
                                "nutrient": nutrient,
                                "delta_g": delta,
                                "actual": gap["actual"],
                                "optimal": gap["target"],
                            }
                        )

                    swap_gap_df = pd.DataFrame(swap_gap_rows)

                    if not swap_gap_df.empty:
                        swap_gap_df["abs_delta"] = swap_gap_df["delta_g"].abs()
                        is_kcal_swap = swap_gap_df["nutrient"].str.contains("kcal", case=False, na=False)

                        non_kcal_swap_df = swap_gap_df[~is_kcal_swap]
                        kcal_swap_df = swap_gap_df[is_kcal_swap]

                        # Chart for non-kcal nutrients after swap
                        if not non_kcal_swap_df.empty:
                            chart_swap = (
                                alt.Chart(non_kcal_swap_df)
                                .mark_bar()
                                .encode(
                                    y=alt.Y(
                                        "nutrient:N",
                                        sort=alt.SortField(field="abs_delta", order="descending"),
                                        title="Nutrient",
                                    ),
                                    x=alt.X(
                                        "delta_g:Q",
                                        title="Difference vs optimal (g)",
                                        scale=alt.Scale(domain=[-50, 50]),
                                    ),
                                    color=alt.value("#22A34F"),
                                    tooltip=[
                                        alt.Tooltip("nutrient:N", title="Nutrient"),
                                        alt.Tooltip("actual:Q", title="Actual (swap)"),
                                        alt.Tooltip("optimal:Q", title="Optimal"),
                                        alt.Tooltip("delta_g:Q", title="Δ vs optimal (g)"),
                                    ],
                                )
                                .properties(height=300)
                            )

                            zero_line_swap = (
                                alt.Chart(pd.DataFrame({"x": [0]}))
                                .mark_rule(strokeDash=[4, 4])
                                .encode(x="x:Q")
                            )

                            st.altair_chart(chart_swap + zero_line_swap, use_container_width=True)
                        else:
                            st.info("No non-kcal nutrient gaps to display for swap.")

                        # Separate handling for kcal after swap
                        if not kcal_swap_df.empty:
                            st.markdown("#### Energy (kcal) after swap")
                            for _, row in kcal_swap_df.iterrows():
                                st.write(
                                    f"**{row['nutrient']}** – Actual (swap): {row['actual']:.0f} kcal, "
                                    f"Optimal: {row['optimal']:.0f} kcal, "
                                    f"Δ: {row['delta_g']:.0f} kcal"
                                )
                    else:
                        st.info("No nutrient gaps to display for swap.")



                    # if not swap_gap_df.empty:
                    #     swap_gap_df["abs_delta"] = swap_gap_df["delta_g"].abs()

                    #     chart_swap = (
                    #         alt.Chart(swap_gap_df)
                    #         .mark_bar()
                    #         .encode(
                    #             y=alt.Y(
                    #                 "nutrient:N",
                    #                 sort=alt.SortField(field="abs_delta", order="descending"),
                    #                 title="Nutrient",
                    #             ),
                    #             x=alt.X(
                    #                 "delta_g:Q",
                    #                 title="Difference vs optimal (g)",
                    #                 scale=alt.Scale(zero=True),
                    #             ),
                    #             color=alt.condition(
                    #                 alt.datum.delta_g > 0,
                    #                 alt.value("#22A34F"),  # above optimal
                    #                 alt.value("#EF4444"),  # below optimal
                    #             ),
                    #             tooltip=[
                    #                 alt.Tooltip("nutrient:N", title="Nutrient"),
                    #                 alt.Tooltip("actual:Q", title="Actual (swap)"),
                    #                 alt.Tooltip("optimal:Q", title="Optimal"),
                    #                 alt.Tooltip("delta_g:Q", title="Δ vs optimal (g)"),
                    #             ],
                    #         )
                    #         .properties(height=300)
                    #     )

                    #     zero_line_swap = (
                    #         alt.Chart(pd.DataFrame({"x": [0]}))
                    #         .mark_rule(strokeDash=[4, 4])
                    #         .encode(x="x:Q")
                    #     )

                    #     st.altair_chart(chart_swap + zero_line_swap, use_container_width=True)
                    # else:
                    #     st.info("No nutrient gaps to display for swap.")

            except Exception as e:
                st.error(f"Error while computing food swap comparison: {e}")

        except Exception as e:
            st.error(f"Error: {e}")
