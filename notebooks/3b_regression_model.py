import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the cleaned numerical data
df = pd.read_csv("data/clean_reclean/numerical_cleaned.csv")

# Drop ID and duplicate SalePrice column if present
df = df.drop(columns=[col for col in df.columns if "Id" in col or col == "SalePrice.1"])

# Drop rows with missing values
df_clean = df.dropna()

# Drop redundant feature: we'll keep GrLivArea and drop TotRmsAbvGrd
X = df_clean.drop(columns=["SalePrice", "TotRmsAbvGrd"])
y = df_clean["SalePrice"]

# Add constant
X_const = sm.add_constant(X)

# Fit initial model
model_initial = sm.OLS(y, X_const).fit()

# Detect outliers: residuals > 3 * std
residuals = model_initial.resid
outliers = np.abs(residuals) > 3 * residuals.std()

# Remove outliers
X_final = X_const[~outliers]
y_final = y[~outliers]

# Fit final model
model_final = sm.OLS(y_final, X_final).fit()

# Show results
print(model_final.summary())

# Optional: Residual stats
print("\nResiduals summary:\n", model_final.resid.describe())
