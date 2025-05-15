import pandas as pd
import numpy as np
import statsmodels.api as sm

# Step 1: Load and clean data
df = pd.read_csv("data/clean_reclean/numerical_cleaned.csv")

# Drop ID columns and duplicate SalePrice column if present
df = df.drop(columns=[col for col in df.columns if "Id" in col or col == "SalePrice.1"])

# Drop rows with missing values
df_clean = df.dropna()

# Step 2: Drop redundant feature
X = df_clean.drop(columns=["SalePrice", "TotRmsAbvGrd"])
y_log = np.log(df_clean["SalePrice"])  # log-transformed target

# Step 3: Fit initial model with all data
X_const = sm.add_constant(X)
log_model_initial = sm.OLS(y_log, X_const).fit()

# Step 4: Detect and remove outliers (log residuals > 3 std)
residuals_log = log_model_initial.resid
outliers = np.abs(residuals_log) > 3 * residuals_log.std()
X_final = X_const[~outliers]
y_final = y_log[~outliers]

# Step 5: Fit final model
log_model_final = sm.OLS(y_final, X_final).fit()

# Step 6: Output summary
print(log_model_final.summary())

# Optional: Residual statistics
print("\nResidual summary:\n", log_model_final.resid.describe())
print("\nFitted value summary:\n", log_model_final.fittedvalues.describe())
