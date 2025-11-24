from nutrimap_app.data_prep_marie import clean_food_data, scale_food_data
from nutrimap_app.KMeanModel import kmeanModel

def build_all():
    df_clean = clean_food_data()
    df_scaled = scale_food_data()
    model, df_clusters = kmeanModel()
    # optionally return or save extra artifacts

if __name__ == "__main__":
    build_all()
