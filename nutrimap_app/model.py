from nutrimap_app.data_prep_final import clean_food_data, scale_food_data
from nutrimap_app.KMeanModel import kmeanModel, subclustering

def build_all():
    df_clean = clean_food_data()
    df_scaled = scale_food_data()

    model, df_clusters = kmeanModel()

    df_with_subclusters = subclustering(df_clusters)

    df_plate_role = assign_plate_role(df_with_subclusters)

    return model, df_plate_role


if __name__ == "__main__":
    build_all()
