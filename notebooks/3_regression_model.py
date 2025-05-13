import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("data/clean_reclean/ordinal_numerized_cleaned.csv")

# Define features
main_features = [
    'KitchenQual_code',
    'GarageFinish_code',
    'BsmtQual_code',
    'ExterQual_code',
    'BsmtExposure_code'
]

interactions = [
    'ExterQual_code:KitchenQual_code',
    'BsmtQual_code:BsmtExposure_code'
]

# Drop rows with missing values
df = df.dropna(subset=main_features + ['SalePrice'])

# ----------------------------------
# 1. Full model with interactions
# ----------------------------------
full_formula = 'SalePrice ~ ' + ' + '.join(main_features + interactions)
model_full = ols(full_formula, data=df).fit()

# ----------------------------------
# 2. Reduced model (main effects only)
# ----------------------------------
reduced_formula = 'SalePrice ~ ' + ' + '.join(main_features)
model_reduced = ols(reduced_formula, data=df).fit()

# ----------------------------------
# 3. Compare models using ANOVA
# ----------------------------------
anova_compare = sm.stats.anova_lm(model_reduced, model_full)
print("ANOVA Comparison (Main effects vs Interactions):\n")
print(anova_compare)

# ----------------------------------
# 4. Summary of the full model
# ----------------------------------
print("\nFull Regression Model Summary:\n")
print(model_full.summary())

# ----------------------------------
# 5. Residual plot
# ----------------------------------
residuals = model_full.resid
fitted = model_full.fittedvalues

plt.figure(figsize=(8, 5))
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.tight_layout()
plt.show()

