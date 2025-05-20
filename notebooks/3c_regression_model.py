import pandas as pd
import numpy as np
import statsmodels.api as sm

# Step 1: Load and clean data
print("Loading and cleaning data...")
df = pd.read_csv("data/clean_reclean/numerical_cleaned.csv")

# Drop ID columns and duplicate SalePrice column if present
df = df.drop(columns=[col for col in df.columns if "Id" in col or col == "SalePrice.1"] )

# Drop rows with missing values
df_clean = df.dropna()

# Step 2: Prepare features and target
print("Preparing features and target...")
X = df_clean.drop(columns=["SalePrice", "TotRmsAbvGrd"])
y_log = np.log(df_clean["SalePrice"])

# Step 3: Fit initial OLS model
print("Fitting initial OLS model...")
X_const = sm.add_constant(X)
log_model_initial = sm.OLS(y_log, X_const).fit()

# Step 4: Detect and remove outliers (|std_resid| > 3)
print("Detecting outliers...")
residuals_log = log_model_initial.resid
outliers = np.abs(residuals_log) > 3 * residuals_log.std()
X_final = X_const[~outliers]
y_final = y_log[~outliers]

# Step 5: Fit final OLS model
print("Fitting final OLS model without outliers...")
log_model_final = sm.OLS(y_final, X_final).fit()

# Step 6: Output summary
print("==== Final OLS Regression Summary ====")
print(log_model_final.summary())
print("\nResidual summary:\n", log_model_final.resid.describe())
print("\nFitted values summary:\n", log_model_final.fittedvalues.describe())
