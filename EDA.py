import pandas as pd
import matplotlib.pyplot as plt

# Loading
def load_split(name):
    return pd.read_csv(f"./data/clean/{name}.csv", index_col=0)

# Load
df_num = pd.read_csv("./data/clean_reclean/numerical_cleaned.csv", index_col=0)
df_cat = pd.read_csv("./data/clean/nominal.csv", index_col=0)
df_ord = pd.read_csv("./data/clean_reclean/ordinal_numerized_cleaned.csv", index_col=0)

print(df_cat[df_cat["Electrical"].isna()].index)
