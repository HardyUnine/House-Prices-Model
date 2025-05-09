import pandas as pd
import matplotlib.pyplot as plt

# Loading
def load_split(name):
    return pd.read_csv(f"./data/clean/{name}.csv", index_col=0)

# Load
df_num = load_split("numeric")
df_cat = load_split("categorical")
df_ord = load_split("ordinal")

