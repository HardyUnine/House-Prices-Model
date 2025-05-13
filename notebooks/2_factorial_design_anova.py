import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/clean_reclean/ordinal_numerized_cleaned.csv")

# Drop ID column if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Define top 6 binary/ordinal features (from previous selection)
features = [
    'ExterQual_code',
    'KitchenQual_code',
    'BsmtQual_code',
    'GarageFinish_code',
    'HeatingQC_code',
    'BsmtExposure_code'
]

# Drop rows with missing SalePrice or selected features (should already be clean, but just in case)
df = df.dropna(subset=features + ['SalePrice'])

# Build formula with main effects + 2-way interactions
main_effects = ' + '.join(features)
interactions = ' + '.join([f"{a}:{b}" for a, b in itertools.combinations(features, 2)])
formula = f"SalePrice ~ {main_effects} + {interactions}"

print(f"Model formula:\n{formula}\n")

# Fit the OLS model
model = ols(formula, data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA Table:\n")
print(anova_table)

# Plot main effects
sns.set(style="whitegrid")
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for i, var in enumerate(features):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=axs[i // 3][i % 3])
    axs[i // 3][i % 3].set_title(f"SalePrice by {var}")
plt.tight_layout()
plt.show()
